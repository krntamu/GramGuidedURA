"""
Diagnostic: NMSE vs reverse step (ℓ) for DM / DM+likelihood guidance.

This script runs the reverse sampling loop once per SNR and records the NMSE at EVERY reverse step,
so we can diagnose whether performance gaps are due to early/mid-step drift or only the last steps.

Run example:

    python tests/test_nmse_vs_step.py \
        --ckpt results/best_models_dm_paper/3gpp_path=3/train_models/model-425.pt \
        --ch_type 3gpp --n_path 3 \
        --snr_list -10,-5,0 \
        --n_test 512 --batch_size 32 \
        --likelihood_mode none \
        --out_dir ./outputs/diagnostics_nmse_step/

Notes:
  - SNR list is given in dB; internally we convert to linear ρ = 10^(SNR/10).
  - We match the repo's evaluation noise model by using DMCE.functional.awgn(..., multiplier=model.noise_multiplier).
  - If the model was trained with FFT pre-processing (sim_params['tester_dict']['fft_pre']=True),
    we do: FFT before inference (network domain) and IFFT before NMSE evaluation (physical domain),
    mirroring DMCE.Tester behavior.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path so `import DMCE` works when running as a script from tests/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import DMCE
from DMCE.utils import cmplx2real
from DMCE import functional
import modules.utils as ut


def parse_snr_list(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if len(vals) == 0:
        raise ValueError("Empty --snr_list")
    return vals


def auto_device(requested: Optional[str]) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def find_sim_params_from_ckpt(ckpt: Path) -> Path:
    """
    Try to infer sim_params.json location from checkpoint path.
    Common layout:
      .../<model_dir>/train_models/model-xxx.pt
      .../<model_dir>/sim_params.json
    """
    candidates = []
    # same directory
    candidates.append(ckpt.parent / "sim_params.json")
    candidates.append(ckpt.parent / "sim_params")
    # parent directory
    candidates.append(ckpt.parent.parent / "sim_params.json")
    candidates.append(ckpt.parent.parent / "sim_params")
    # grandparent directory
    candidates.append(ckpt.parent.parent.parent / "sim_params.json")
    candidates.append(ckpt.parent.parent.parent / "sim_params")

    for c in candidates:
        if c.suffix == ".json" and c.is_file():
            return c
        if c.suffix == "" and (c.with_suffix(".json")).is_file():
            return c.with_suffix(".json")
    raise FileNotFoundError(
        "Could not infer sim_params.json from ckpt path. "
        "Expected sim_params.json near the checkpoint directory."
    )


def load_sim_params(sim_params_path: Path) -> dict:
    # DMCE.utils.load_params expects base path without suffix (it appends .json)
    base = sim_params_path
    if base.suffix == ".json":
        base = base.with_suffix("")
    return DMCE.utils.load_params(str(base))


def load_dm_from_ckpt(ckpt: Path, sim_params: dict, device: torch.device) -> DMCE.DiffusionModel:
    cnn_dict = sim_params["unet_dict"].copy()
    diff_model_dict = sim_params["diff_model_dict"].copy()
    cnn_dict["device"] = str(device)

    cnn = DMCE.CNN(**cnn_dict)
    dm = DMCE.DiffusionModel(cnn, **diff_model_dict)

    state = torch.load(str(ckpt), map_location=str(device))
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    dm.load_state_dict(sd, strict=False)
    dm.to(device=device)
    dm.eval()
    return dm


def prep_test_data(
    *,
    ch_type: str,
    n_path: int,
    n_rx: int,
    n_tx: int,
    n_train: int,
    n_val: int,
    n_test: int,
    device: torch.device,
) -> torch.Tensor:
    def _infer_available_test_sizes_3gpp() -> List[int]:
        """
        Look for files like:
          bin/3gpp_path=3_dimrx=64_dimtx=16_samp=10000_test.npy
        and return available samp sizes as ints.
        """
        bin_dir = _REPO_ROOT / "bin"
        if not bin_dir.is_dir():
            return []
        pat = f"3gpp_path={n_path}_dimrx={n_rx}_dimtx={n_tx}_samp=*_*test.npy"
        # be permissive about exact suffix, but require "_test.npy"
        files = list(bin_dir.glob(f"3gpp_path={n_path}_dimrx={n_rx}_dimtx={n_tx}_samp=*_test.npy"))
        sizes: List[int] = []
        for fp in files:
            name = fp.name
            # parse "..._samp=NNN_test.npy"
            try:
                left = name.split("_samp=")[1]
                n_str = left.split("_test.npy")[0]
                sizes.append(int(n_str))
            except Exception:
                continue
        return sorted(set(sizes))

    def _pick_test_size(requested: int) -> int:
        if not ch_type.startswith("3gpp") or n_tx <= 1:
            return requested
        sizes = _infer_available_test_sizes_3gpp()
        if not sizes:
            return requested
        # Prefer smallest available >= requested, else max available
        for s in sizes:
            if s >= requested:
                return s
        return sizes[-1]

    n_test_load = _pick_test_size(n_test)

    # Load complex channel samples (numpy)
    try:
        _, _, data_test = ut.load_or_create_data(
            ch_type=ch_type,
            n_path=n_path,
            n_antennas_rx=n_rx,
            n_antennas_tx=n_tx,
            n_train_ch=n_train,
            n_val_ch=n_val,
            n_test_ch=n_test_load,
            return_toep=False,
        )
    except FileNotFoundError as e:
        # Common case for 3gpp: user requests a smaller n_test than what's available on disk.
        # We rethrow with a more actionable message.
        raise FileNotFoundError(
            f"{e}\n"
            f"[Hint] Your bin/ directory likely contains a fixed test set size (e.g. 10000). "
            f"This script will load the closest available test set and then subsample to --n_test, "
            f"but it couldn't find any compatible files for ch_type={ch_type}, n_path={n_path}, n_rx={n_rx}, n_tx={n_tx}."
        ) from e

    # 3GPP MIMO stored vectorized; reshape to (N, n_rx, n_tx) Fortran order
    if ch_type.startswith("3gpp") and n_tx > 1:
        data_test = np.reshape(data_test, (-1, n_rx, n_tx), order="F")

    # To torch with explicit channel dim (complex -> 2 real channels)
    # Subsample to requested n_test (after loading)
    if n_test < data_test.shape[0]:
        data_test = data_test[:n_test]

    data_test_t = torch.from_numpy(np.asarray(data_test[:, None, :]))
    # Keep dataset on CPU for DataLoader; move batches to device inside the loop.
    data_test_t = cmplx2real(data_test_t, dim=1, new_dim=False).float()
    return data_test_t


def maybe_fft_preprocess(data: torch.Tensor, *, fft_pre: bool, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mirror DMCE.Tester behavior:
      - data_net: what the network sees (FFT applied if fft_pre=True)
      - data_eval: what we use for NMSE (IFFT applied if fft_pre=True)
    """
    if not fft_pre:
        return data, data
    data_net = ut.complex_1d_fft(data, ifft=False, mode=mode)
    data_eval = ut.complex_1d_fft(data_net, ifft=True, mode=mode)
    return data_net, data_eval


def nmse_per_sample_per_step(x_steps: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """
    x_steps: (B, L, *shape)
    x_true:  (B, *shape)
    returns: (B, L) NMSE per sample per step
    """
    B, L = x_steps.shape[0], x_steps.shape[1]
    xh = x_steps.reshape(B, L, -1)
    xt = x_true.reshape(B, -1)
    num = torch.sum((xh - xt[:, None, :]) ** 2, dim=-1)
    den = torch.sum(xt**2, dim=-1).clamp_min(1e-12)
    return num / den[:, None]


def stats_over_samples(nmse_sl: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    nmse_sl: (N, L) on CPU
    returns mean/median/p10/p90 arrays (L,)
    """
    nmse_np = nmse_sl.numpy()
    return {
        "mean": np.nanmean(nmse_np, axis=0),
        "median": np.nanmedian(nmse_np, axis=0),
        "p10": np.nanpercentile(nmse_np, 10, axis=0),
        "p90": np.nanpercentile(nmse_np, 90, axis=0),
    }


def summarize_curve(y: np.ndarray) -> dict:
    """
    Summarize a mean-NMSE curve (shape [L]).
    """
    idx = np.where(np.isfinite(y))[0]
    if idx.size == 0:
        return {"final_step": None, "final_nmse": None, "first_within_10pct_final": None}
    final_step = int(idx[-1])
    final_nmse = float(y[final_step])
    thr = final_nmse * 1.10
    within = np.where(np.isfinite(y) & (y <= thr))[0]
    first_within = int(within[0]) if within.size else None
    return {
        "final_step": final_step,
        "final_nmse": final_nmse,
        "first_within_10pct_final": first_within,
    }


def compare_curves(base: np.ndarray, other: np.ndarray, k_stable: int = 10) -> dict:
    """
    Compare two mean curves. Positive improvement means `other` is better (lower NMSE).
    """
    mask = np.isfinite(base) & np.isfinite(other)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return {"max_improvement": None, "step_max_improvement": None, "first_stable_win": None, "k_stable": int(k_stable)}
    diff = base[idx] - other[idx]
    jmax = int(np.argmax(diff))
    max_impr = float(diff[jmax])
    step_max = int(idx[jmax])
    first_stable = None
    if diff.size >= k_stable:
        for j in range(0, diff.size - k_stable + 1):
            if np.all(diff[j : j + k_stable] > 0):
                first_stable = int(idx[j])
                break
    return {
        "max_improvement": max_impr,
        "step_max_improvement": step_max,
        "first_stable_win": first_stable,
        "k_stable": int(k_stable),
    }


def plot_nmse_vs_step(
    *,
    out_path: Path,
    steps: np.ndarray,
    curves: Dict[str, Dict[str, np.ndarray]],
    snr_db: float,
    title_prefix: str = "",
) -> None:
    import matplotlib
    if "DISPLAY" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa

    plt.figure(figsize=(8.5, 5.5))
    for name, stats in curves.items():
        y = stats["mean"]
        m = np.isfinite(y)
        plt.semilogy(steps[m], y[m], label=name, linewidth=2.0)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.xlabel("Reverse step ℓ (0=start, larger=more denoised)")
    plt.ylabel("NMSE (mean over samples)")
    plt.title(f"{title_prefix}NMSE vs step @ SNR={snr_db:.1f} dB")
    plt.legend(loc="best", framealpha=0.95)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnostic: NMSE vs reverse step (DM / likelihood-guided)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to DM checkpoint (.pt)")
    p.add_argument("--ch_type", type=str, default="3gpp", choices=["3gpp", "quadriga_LOS"])
    p.add_argument("--n_path", type=int, default=3)
    p.add_argument("--snr_list", type=str, default="-10,-5,0", help="Comma-separated SNRs in dB, e.g. -10,-5,0")
    p.add_argument("--n_test", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./outputs/diagnostics_nmse_step/")
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional tag to separate outputs (used as a subfolder under --out_dir). "
             "If not set, a timestamp+config tag is generated automatically.",
    )
    g_npz = p.add_mutually_exclusive_group()
    g_npz.add_argument("--save_npz", dest="save_npz", action="store_true", help="Save raw arrays to npz (default: on)")
    g_npz.add_argument("--no_save_npz", dest="save_npz", action="store_false", help="Disable saving nmse_vs_step.npz")
    p.set_defaults(save_npz=True)
    p.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")

    # optional / extended knobs
    g_snr = p.add_mutually_exclusive_group()
    g_snr.add_argument("--use_snr_cond", dest="use_snr_cond", action="store_true", help="Use SNR-matched start timestep (t_start)")
    g_snr.add_argument("--no_use_snr_cond", dest="use_snr_cond", action="store_false", help="Disable SNR matching (use full reverse chain)")
    p.set_defaults(use_snr_cond=True)
    p.add_argument("--likelihood_mode", type=str, default="none", choices=["none", "closed_form", "tweedie"])
    p.add_argument("--lambda_lik", type=float, default=0.1, help="Likelihood guidance strength")
    p.add_argument("--L", type=int, default=None, help="Optional maximum number of steps to record (truncate/pad to this length).")
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val", type=int, default=10_000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {ckpt}")

    device = auto_device(args.device)
    print(f"[Setup] device={device}")

    sim_params_path = find_sim_params_from_ckpt(ckpt)
    sim_params = load_sim_params(sim_params_path)
    print(f"[Setup] sim_params={sim_params_path}")

    dm = load_dm_from_ckpt(ckpt, sim_params, device=device)
    print(f"[Setup] DM num_timesteps={dm.num_timesteps}, reverse_add_random={dm.reverse_add_random}")

    # Data shape from sim_params (preferred), fallback to common defaults
    data_shape = sim_params.get("data_dict", {}).get("data_shape", sim_params.get("diff_model_dict", {}).get("data_shape"))
    if data_shape is None:
        raise ValueError("Could not infer data_shape from sim_params (data_dict.data_shape or diff_model_dict.data_shape).")
    # expected like [2, 64, 16]
    if len(data_shape) < 3:
        raise ValueError(f"Expected 2D channel data_shape like [2, n_rx, n_tx], got {data_shape}")
    n_rx = int(data_shape[-2])
    n_tx = int(data_shape[-1])

    mode = sim_params.get("data_dict", {}).get("mode", sim_params.get("tester_dict", {}).get("mode", "2D"))
    fft_pre = bool(sim_params.get("tester_dict", {}).get("fft_pre", True))
    print(f"[Setup] data_shape={data_shape}, n_rx={n_rx}, n_tx={n_tx}, mode={mode}, fft_pre={fft_pre}")

    # Load test data (complex -> real channels)
    data_test = prep_test_data(
        ch_type=args.ch_type,
        n_path=args.n_path,
        n_rx=n_rx,
        n_tx=n_tx,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        device=device,
    )
    # Keep both tensors on CPU for DataLoader; move per-batch to device.
    data_net, data_eval = maybe_fft_preprocess(data_test, fft_pre=fft_pre, mode=mode)

    # Dataloader uses network-domain samples
    loader = torch.utils.data.DataLoader(data_net, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    snr_db_list = parse_snr_list(args.snr_list)
    snr_lin_list = [10 ** (s / 10.0) for s in snr_db_list]

    # Determine max number of steps to record
    L_max = int(args.L) if args.L is not None else int(dm.num_timesteps)
    steps_axis = np.arange(L_max, dtype=int)

    # Methods to run
    methods: List[str] = ["DM (+SNR)" if args.use_snr_cond else "DM (fullT)"]
    if args.use_snr_cond:
        methods.append("DM (fullT)")  # contrast curve
    if args.likelihood_mode != "none":
        methods.append(f"DM + likelihood ({args.likelihood_mode})")
    # de-dup
    methods = list(dict.fromkeys([m for m in methods if m is not None]))

    # Build DPS sampler if needed
    dps_sampler = None
    if args.likelihood_mode != "none":
        from dps_sampler import DpsSampler, make_awgn_likelihood_grad  # local import
        exp_key = "E" if args.likelihood_mode == "closed_form" else "H"
        # sigma_y2 depends on SNR; we build likelihood_grad_fn per SNR in the loop
        dps_sampler = (DpsSampler, make_awgn_likelihood_grad, exp_key)

    # Collect results
    nmse_mean = {m: np.full((len(snr_db_list), L_max), np.nan, dtype=float) for m in methods}
    nmse_median = {m: np.full((len(snr_db_list), L_max), np.nan, dtype=float) for m in methods}
    nmse_p10 = {m: np.full((len(snr_db_list), L_max), np.nan, dtype=float) for m in methods}
    nmse_p90 = {m: np.full((len(snr_db_list), L_max), np.nan, dtype=float) for m in methods}

    def _sanitize_tag(s: str) -> str:
        # Keep filenames safe and short-ish
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-=+"
        s2 = "".join([c if c in allowed else "_" for c in s])
        # collapse repeats
        while "__" in s2:
            s2 = s2.replace("__", "_")
        return s2.strip("_")

    if args.run_tag is None:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = f"{ts}__snrcond={int(args.use_snr_cond)}__lik={args.likelihood_mode}__lam={args.lambda_lik:g}__seed={args.seed}"
    else:
        tag = args.run_tag
    tag = _sanitize_tag(tag)

    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] run_tag={tag}")
    print(f"[Output] out_dir={out_dir}")

    # Human-readable summary (JSON) so we can analyze runs without parsing npz/images.
    summary: dict = {
        "run_tag": tag,
        "ckpt": str(ckpt),
        "sim_params": str(sim_params_path),
        "device": str(device),
        "ch_type": args.ch_type,
        "n_path": int(args.n_path),
        "snr_list_db": snr_db_list,
        "n_test": int(args.n_test),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "use_snr_cond": bool(args.use_snr_cond),
        "likelihood_mode": args.likelihood_mode,
        "lambda_lik": float(args.lambda_lik),
        "L": int(L_max),
        "fft_pre": bool(fft_pre),
        "mode": str(mode),
        "methods": methods,
        "per_snr": {},
    }

    for si, (snr_db, snr_lin) in enumerate(zip(snr_db_list, snr_lin_list)):
        print(f"\n[SNR {snr_db:.1f} dB] rho={snr_lin:.4g}")

        # Per-method per-sample curves (concatenated across batches)
        per_method_nmse_samples: Dict[str, List[torch.Tensor]] = {m: [] for m in methods}

        # For guided sampling, build a per-SNR sampler with correct sigma_y2
        if dps_sampler is not None:
            DpsSampler, make_awgn_likelihood_grad, exp_key = dps_sampler
            noise_mult = float(dm.noise_multiplier)
            sigma_y2 = (noise_mult ** 2) / float(snr_lin)
            like_grad_fn = make_awgn_likelihood_grad(sigma_y2)
            sampler = DpsSampler(
                dm=dm,
                likelihood_grad_fn=like_grad_fn,
                dps_lambda=float(args.lambda_lik),
                cov_lambda=0.0,
                add_random=bool(dm.reverse_add_random),
                exp_key=exp_key,
                sigma_y2=float(sigma_y2),
            )
        else:
            sampler = None

        # Precompute t_hat for SNR-matched DM (+SNR)
        t_hat = int(torch.abs(dm.snrs - torch.tensor(float(snr_lin), device=dm.device)).argmin().item())
        t_hat = min(t_hat, L_max - 1)

        for batch_idx, x0_net in enumerate(loader):
            # x0_net is CPU (optionally pinned); move to device here
            x0_net = x0_net.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
            # match evaluation ground truth for this batch
            start = batch_idx * args.batch_size
            end = min(start + x0_net.shape[0], data_eval.shape[0])
            x0_eval = data_eval[start:end].to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))

            # Corrupt in network domain
            y = functional.awgn(x0_net, float(snr_lin), multiplier=float(dm.noise_multiplier))

            # Helper: convert x_steps to eval domain if needed
            def to_eval_domain(x_steps: torch.Tensor) -> torch.Tensor:
                if not fft_pre:
                    return x_steps
                # x_steps is (B, L, 2, n_rx, n_tx) when return_all_timesteps
                return ut.complex_1d_fft(x_steps, ifft=True, mode=mode, _4d_array=True)

            # --- DM (+SNR) or DM (fullT) ---
            if args.use_snr_cond:
                x_steps = dm.generate_estimate(y, float(snr_lin), return_all_timesteps=True)
                # generate_estimate returns length (t_hat+1); we still truncate/pad later
            else:
                norm_multiplier = math.sqrt(float(snr_lin) / (1.0 + float(snr_lin)))
                x_t = norm_multiplier * y
                x_steps = dm.reverse_sample_loop(x_t, int(L_max - 1), return_all_timesteps=True)

            x_steps_eval = to_eval_domain(x_steps)
            nmse_sl = nmse_per_sample_per_step(x_steps_eval, x0_eval)  # (B, L_local)
            if torch.any(~torch.isfinite(nmse_sl)):
                raise RuntimeError(f"NaN/Inf NMSE detected for DM at SNR={snr_db}dB in batch {batch_idx}")
            per_method_nmse_samples[methods[0]].append(nmse_sl.detach().cpu())

            # Optional contrast: DM (fullT) when use_snr_cond is enabled
            if args.use_snr_cond and "DM (fullT)" in per_method_nmse_samples:
                norm_multiplier = math.sqrt(float(snr_lin) / (1.0 + float(snr_lin)))
                x_t = norm_multiplier * y
                x_steps2 = dm.reverse_sample_loop(x_t, int(L_max - 1), return_all_timesteps=True)
                x_steps2_eval = to_eval_domain(x_steps2)
                nmse2 = nmse_per_sample_per_step(x_steps2_eval, x0_eval)
                if torch.any(~torch.isfinite(nmse2)):
                    raise RuntimeError(f"NaN/Inf NMSE detected for DM(fullT) at SNR={snr_db}dB in batch {batch_idx}")
                per_method_nmse_samples["DM (fullT)"].append(nmse2.detach().cpu())

            # --- DM + likelihood guidance ---
            if sampler is not None:
                cov_zeros = torch.zeros((x0_net.shape[0], 2, n_rx, n_rx), device=device, dtype=torch.float32)
                norm_multiplier = math.sqrt(float(snr_lin) / (1.0 + float(snr_lin)))
                x_T = norm_multiplier * y  # initialize from observation even if we disable SNR-matching

                snr_for_loop = float(snr_lin) if args.use_snr_cond else None
                x_steps3 = sampler.generate_posterior_sample(
                    y,
                    cov=cov_zeros,
                    x_T=x_T,
                    return_all_timesteps=True,
                    num_steps=int(L_max),
                    snr=snr_for_loop,
                    obs_snr_db=float(snr_db),
                )
                x_steps3_eval = to_eval_domain(x_steps3)
                nmse3 = nmse_per_sample_per_step(x_steps3_eval, x0_eval)
                if torch.any(~torch.isfinite(nmse3)):
                    raise RuntimeError(f"NaN/Inf NMSE detected for guided at SNR={snr_db}dB in batch {batch_idx}")
                per_method_nmse_samples[f"DM + likelihood ({args.likelihood_mode})"].append(nmse3.detach().cpu())

        # Aggregate & pad/truncate to L_max
        curves_for_plot: Dict[str, Dict[str, np.ndarray]] = {}
        for m in methods:
            nmse_cat = torch.cat(per_method_nmse_samples[m], dim=0)  # (N, L_local)
            # pad/truncate
            L_local = nmse_cat.shape[1]
            if L_local < L_max:
                pad = torch.full((nmse_cat.shape[0], L_max - L_local), float("nan"))
                nmse_cat = torch.cat([nmse_cat, pad], dim=1)
            elif L_local > L_max:
                nmse_cat = nmse_cat[:, :L_max]

            st = stats_over_samples(nmse_cat)
            nmse_mean[m][si, :] = st["mean"]
            nmse_median[m][si, :] = st["median"]
            nmse_p10[m][si, :] = st["p10"]
            nmse_p90[m][si, :] = st["p90"]
            curves_for_plot[m] = st

            # Sanity print final step (last finite)
            finite = np.isfinite(st["mean"])
            last_idx = int(np.where(finite)[0][-1]) if np.any(finite) else -1
            final_nmse = st["mean"][last_idx] if last_idx >= 0 else float("nan")
            print(f"  {m:>22s}: final_step={last_idx}  NMSE_mean={final_nmse:.6g}")

        # Per-SNR summary (vs baseline)
        base = methods[0]
        snr_key = f"{snr_db:+.1f}dB"
        per = {
            "snr_db": float(snr_db),
            "rho": float(snr_lin),
            "base_method": base,
            "methods": {m: summarize_curve(nmse_mean[m][si, :]) for m in methods},
            "comparisons_vs_base": {m: compare_curves(nmse_mean[base][si, :], nmse_mean[m][si, :], k_stable=10) for m in methods[1:]},
        }
        summary["per_snr"][snr_key] = per

        # Per-SNR plot
        plot_nmse_vs_step(
            out_path=out_dir / f"nmse_vs_step__snr_{snr_db:+.1f}dB.png",
            steps=steps_axis,
            curves=curves_for_plot,
            snr_db=float(snr_db),
            title_prefix="",
        )

    # Save NPZ
    if args.save_npz:
        out_npz = out_dir / "nmse_vs_step.npz"
        payload = {
            "snr_list_db": np.array(snr_db_list, dtype=float),
            "steps": steps_axis,
            "methods": np.array(methods),
            "run_tag": np.array([tag]),
            "use_snr_cond": np.array([int(args.use_snr_cond)]),
            "likelihood_mode": np.array([args.likelihood_mode]),
            "lambda_lik": np.array([float(args.lambda_lik)]),
            "seed": np.array([int(args.seed)]),
        }
        for m in methods:
            key = m.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("-", "minus")
            payload[f"nmse_mean__{key}"] = nmse_mean[m]
            payload[f"nmse_median__{key}"] = nmse_median[m]
            payload[f"nmse_p10__{key}"] = nmse_p10[m]
            payload[f"nmse_p90__{key}"] = nmse_p90[m]
        np.savez(out_npz, **payload)
        print(f"\n[saved] {out_npz}")

    # Save JSON summary (human readable)
    out_json = out_dir / "nmse_vs_step_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_json}")

    # Combined plot (all SNRs, linestyle per method)
    import matplotlib
    if "DISPLAY" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa

    plt.figure(figsize=(9.0, 6.0))
    colors = plt.cm.viridis(np.linspace(0, 1, len(snr_db_list)))
    linestyles = ["-", "--", "-.", ":"]
    for mi, m in enumerate(methods):
        ls = linestyles[mi % len(linestyles)]
        for si, snr_db in enumerate(snr_db_list):
            y = nmse_mean[m][si, :]
            mask = np.isfinite(y)
            plt.semilogy(
                steps_axis[mask],
                y[mask],
                color=colors[si],
                linestyle=ls,
                linewidth=1.8,
                label=f"{m} @ {snr_db:+.1f}dB" if mi == 0 else None,  # avoid legend explosion
                alpha=0.9,
            )
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.xlabel("Reverse step ℓ")
    plt.ylabel("NMSE (mean)")
    plt.title("NMSE vs step (all SNRs)")
    # Legend: only SNR colors (first method only)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc="best", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    out_png = out_dir / "nmse_vs_step__all_snrs.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()


