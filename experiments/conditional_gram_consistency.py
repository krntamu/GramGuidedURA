"""
Conditional diagnostic experiment (A2'/B1): fixed observation y, compare:
  1) likelihood-only guidance
  2) Gram-only (cov) guidance
  3) Gram + likelihood guidance

Focus: QuaDRiGa (LOS) case, but works for 3GPP too if data exists.

We condition on the SAME y generated from a single ground-truth H* at a chosen SNR.
For each mode, run multiple seeds and collect:
  - data consistency e_y (AWGN residual energy, aligned with likelihood)
  - Gram consistency e_G (trace-normalized Gram distance to G*)
  - NMSE vs H*

Outputs:
  - scatter: e_y vs e_G
  - boxplots: e_y, e_G, NMSE per mode
  - saved metrics (.npz + .csv) under results/useful_results/

Important:
  - Observation operator here matches the evaluation pipeline: y = awgn(H*, snr_db).
    i.e., A = Identity in the diffusion domain used by the sampler.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np

import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import DMCE
from DMCE import functional
from DMCE.utils import cmplx2real
import modules.utils as ut

from dps_sampler import DpsSampler, make_awgn_likelihood_grad
from load_and_eval_dm_dps import estimate_cov_time_averaged_batch, generate_cov_batch


EPS = 1e-12


def _to_complex(x_ri: torch.Tensor) -> torch.Tensor:
    """x_ri: (B,2,Nr,Nt) -> complex (B,Nr,Nt)"""
    return torch.complex(x_ri[:, 0], x_ri[:, 1])

def _cov_ri_to_complex(cov_ri: torch.Tensor) -> torch.Tensor:
    """cov_ri: (B,2,Nr,Nr) -> complex (B,Nr,Nr)"""
    return torch.complex(cov_ri[:, 0], cov_ri[:, 1])


def compute_gram_complex(Hc: torch.Tensor) -> torch.Tensor:
    """Hc: (B,Nr,Nt) complex -> G: (B,Nr,Nr) complex"""
    return Hc @ Hc.conj().transpose(-1, -2)


def trace_normalize_gram(G: torch.Tensor) -> torch.Tensor:
    tr = torch.real(torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1))
    tr = torch.clamp(tr, min=EPS)
    return G / tr[:, None, None]


def frob_norm_complex(A: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.abs(A) ** 2, dim=(-2, -1)))


def compute_data_error_awgn(H_hat_ri: torch.Tensor, y_ri: torch.Tensor, sigma_y2: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      - mse_per_sample: mean(|y - H_hat|^2) per complex entry, shape (B,)
      - nll_like_per_sample: (1/sigma_y2) * sum(|y - H_hat|^2), shape (B,)
    """
    res = _to_complex(y_ri) - _to_complex(H_hat_ri)
    sq = torch.abs(res) ** 2
    red_dims = tuple(range(1, sq.ndim))
    mse = torch.mean(sq, dim=red_dims)
    nll = torch.sum(sq, dim=red_dims) / float(sigma_y2)
    return mse, nll


def maybe_fft_pre(x_ri: torch.Tensor, *, fft_pre: bool, mode: str) -> torch.Tensor:
    """
    Mirror DMCE.Tester behavior:
      - if fft_pre=True, the model input domain is ut.complex_1d_fft(x, ifft=False)
    """
    if not fft_pre:
        return x_ri
    return ut.complex_1d_fft(x_ri, ifft=False, mode=mode)


def maybe_ifft_post(x_ri: torch.Tensor, *, fft_pre: bool, mode: str) -> torch.Tensor:
    """
    Mirror evaluation post-processing in load_and_eval_dm_dps.py:
      - if fft_pre=True, estimates are transformed back with ifft=True for NMSE.
    """
    if not fft_pre:
        return x_ri
    return ut.complex_1d_fft(x_ri, ifft=True, mode=mode)


def compute_nmse_dmce_style(H_hat_ri: torch.Tensor, H_gt_ri: torch.Tensor) -> torch.Tensor:
    """
    Match DMCE.functional.nmse_torch(..., norm_per_sample=False):
      nmse = mean( ||x - x_hat||^2 / mean(||x||^2) )

    Returns per-sample contributions numerator/mean_den, shape (B,).
    """
    assert H_hat_ri.shape == H_gt_ri.shape
    B = H_gt_ri.shape[0]
    x = H_gt_ri.reshape(B, -1)
    xh = H_hat_ri.reshape(B, -1)
    num = torch.linalg.vector_norm(x - xh, dim=1) ** 2
    den = torch.linalg.vector_norm(x, dim=1) ** 2
    den_mean = torch.mean(den).clamp(min=EPS)
    return num / den_mean


def compute_gram_error(H_hat_ri: torch.Tensor, G_ref_tilde: torch.Tensor) -> torch.Tensor:
    Hc = _to_complex(H_hat_ri)
    G = compute_gram_complex(Hc)
    Gt = trace_normalize_gram(G)
    d = frob_norm_complex(Gt - G_ref_tilde)
    return d


def prepare_split_numpy(
    *,
    ch_type: str,
    n_rx: int,
    n_tx: int,
    n_path: int,
    n_train: int,
    n_val: int,
    n_test: int,
    split: str,
    start_idx: int,
    n_samples: int,
) -> Tuple[np.ndarray, str]:
    """
    Returns:
      H_split: complex ndarray, shape (N, n_rx, n_tx)
      ch_type_tag: for filenames
    """
    data_train, data_val, data_test = ut.load_or_create_data(
        ch_type=ch_type,
        n_path=n_path,
        n_antennas_rx=n_rx,
        n_antennas_tx=n_tx,
        n_train_ch=n_train,
        n_val_ch=n_val,
        n_test_ch=n_test,
        return_toep=False,
    )
    if split == "train":
        data_split = data_train
    elif split == "val":
        data_split = data_val
    elif split == "test":
        data_split = data_test
    else:
        raise ValueError("split must be one of: train, val, test")

    if ch_type.startswith("3gpp") and n_tx > 1:
        data_split = np.reshape(data_split, (-1, n_rx, n_tx), order="F")

    if n_tx == 1 and data_split.ndim == 2:
        data_split = data_split[..., None]

    end_idx = start_idx + n_samples
    if not (0 <= start_idx < data_split.shape[0]) or end_idx > data_split.shape[0]:
        raise ValueError(
            f"Requested range [{start_idx},{end_idx}) out of bounds for split={split} with size={data_split.shape[0]}"
        )

    H_split = np.asarray(data_split[start_idx:end_idx])
    ch_type_tag = f"{ch_type}_path={n_path}" if ch_type.startswith("3gpp") else ch_type
    return H_split, ch_type_tag


def complex_numpy_to_ri_torch(Hc: np.ndarray) -> torch.Tensor:
    """
    Hc: complex ndarray (B, Nr, Nt) -> (B, 2, Nr, Nt) float32
    """
    x = torch.from_numpy(np.asarray(Hc)[:, None, ...])  # (B,1,Nr,Nt)
    x = cmplx2real(x, dim=1, new_dim=False).float()     # (B,2,Nr,Nt)
    return x


@torch.no_grad()
def debug_gram_reference_consistency(
    *,
    H_star_in: torch.Tensor,
    H_star_eval: torch.Tensor,
    cov_ri: torch.Tensor,
) -> Dict[str, float]:
    """
    Implements the 2 sanity checks suggested in your screenshot:

    V1) Are we guiding toward the same Gram we use to evaluate e_G?
        Compare trace-normalized Gram from guidance reference (cov) vs GT Gram.

    V2) Does e_G depend strongly on which domain we compute the GT Gram in?
        Compare GT Gram computed from H_star_in vs H_star_eval (IFFT domain).
    """
    # Guidance reference Gram (from cov)
    G_guide = _cov_ri_to_complex(cov_ri)
    G_guide_t = trace_normalize_gram(G_guide)

    # GT Gram in model input domain
    G_star_in = compute_gram_complex(_to_complex(H_star_in))
    G_star_in_t = trace_normalize_gram(G_star_in)

    # GT Gram in eval domain
    G_star_eval = compute_gram_complex(_to_complex(H_star_eval))
    G_star_eval_t = trace_normalize_gram(G_star_eval)

    d_guide_vs_star_in = frob_norm_complex(G_guide_t - G_star_in_t).mean().item()
    d_guide_vs_star_eval = frob_norm_complex(G_guide_t - G_star_eval_t).mean().item()
    d_star_in_vs_eval = frob_norm_complex(G_star_in_t - G_star_eval_t).mean().item()

    return {
        "d_Gguide_vs_Gstar_in_trace": float(d_guide_vs_star_in),
        "d_Gguide_vs_Gstar_eval_trace": float(d_guide_vs_star_eval),
        "d_Gstar_in_vs_eval_trace": float(d_star_in_vs_eval),
    }


def prepare_single_sample(
    *,
    ch_type: str,
    n_rx: int,
    n_tx: int,
    n_path: int,
    n_train: int,
    n_val: int,
    n_test: int,
    split: str,
    sample_idx: int,
) -> Tuple[torch.Tensor, str]:
    """
    Returns:
      H_star_ri: (1,2,n_rx,n_tx) float tensor
      ch_type_tag: for filenames
    """
    data_train, data_val, data_test = ut.load_or_create_data(
        ch_type=ch_type,
        n_path=n_path,
        n_antennas_rx=n_rx,
        n_antennas_tx=n_tx,
        n_train_ch=n_train,
        n_val_ch=n_val,
        n_test_ch=n_test,
        return_toep=False,
    )
    if split == "train":
        data_split = data_train
    elif split == "val":
        data_split = data_val
    elif split == "test":
        data_split = data_test
    else:
        raise ValueError("split must be one of: train, val, test")

    if ch_type.startswith("3gpp") and n_tx > 1:
        data_split = np.reshape(data_split, (-1, n_rx, n_tx), order="F")

    if not (0 <= sample_idx < data_split.shape[0]):
        raise ValueError(f"sample_idx={sample_idx} out of range [0,{data_split.shape[0]-1}] for split={split}")

    Hc = np.asarray(data_split[sample_idx])[None, None, ...]  # (1,1,Nr,Nt) complex-ish
    H_ri = torch.from_numpy(Hc)
    H_ri = cmplx2real(H_ri, dim=1, new_dim=False).float()  # (1,2,Nr,Nt)

    ch_type_tag = f"{ch_type}_path={n_path}" if ch_type.startswith("3gpp") else ch_type
    return H_ri, ch_type_tag


def load_diffusion_model(model_dir: Path, device: str) -> DMCE.DiffusionModel:
    """
    model_dir should point to a directory containing:
      - sim_params.json (or sim_params)
      - train_models/ with model-*.pt checkpoints

    For convenience, we also accept:
      - a path directly to sim_params.json
      - a parent directory, in which case we try to locate sim_params.json under it
    """
    model_dir = Path(model_dir)

    # Accept passing sim_params(.json) directly
    if model_dir.is_file() and model_dir.name.startswith("sim_params") and model_dir.suffix == ".json":
        sim_params_path = model_dir
        model_dir = model_dir.parent
    else:
        # Preferred: model_dir/sim_params.json
        sim_params_path = model_dir / "sim_params.json"
        if not sim_params_path.exists():
            # Try model_dir/sim_params (DMCE.utils.load_params appends .json)
            if (model_dir / "sim_params").exists():
                sim_params_path = model_dir / "sim_params"
            else:
                # Search one level deep for sim_params.json
                candidates = list(model_dir.glob("**/sim_params.json"))
                if len(candidates) == 1:
                    sim_params_path = candidates[0]
                    model_dir = sim_params_path.parent
                else:
                    cand_str = "\n".join([f"  - {c}" for c in candidates[:10]]) if candidates else "  (none found)"
                    raise ValueError(
                        "Could not find sim_params.json under the provided --model_dir.\n"
                        f"Provided: {model_dir}\n"
                        "Expected either:\n"
                        "  - <model_dir>/sim_params.json\n"
                        "  - <model_dir>/train_models/model-*.pt\n"
                        "Or pass sim_params.json directly.\n"
                        f"Search candidates (up to 10):\n{cand_str}"
                    )

    # DMCE.utils.load_params expects a base path without .json (it appends if missing)
    sim_params = DMCE.utils.load_params(str(sim_params_path.with_suffix("")))
    cnn_dict = sim_params["unet_dict"]
    diff_model_dict = sim_params["diff_model_dict"]

    cnn_dict["device"] = device
    cnn = DMCE.CNN(**cnn_dict)
    diffusion_model = DMCE.DiffusionModel(cnn, **diff_model_dict)

    ckpt_dir = model_dir / "train_models"
    if not ckpt_dir.is_dir():
        raise ValueError(f"Could not find checkpoints dir: {ckpt_dir}")
    ckpts = sorted(os.listdir(ckpt_dir))
    if len(ckpts) == 0:
        raise ValueError(f"No checkpoints found under: {ckpt_dir}")
    checkpoint = ckpt_dir / ckpts[-1]
    model_state = torch.load(checkpoint, map_location=device)
    diffusion_model.load_state_dict(model_state["model"])
    diffusion_model.eval()
    return diffusion_model


@dataclass
class RunConfig:
    name: str
    dps_lambda: float
    cov_lambda: float


def run_mode(
    *,
    cfg: RunConfig,
    dm: DMCE.DiffusionModel,
    H_star_ri: torch.Tensor,
    y_ri: torch.Tensor,
    cov_ri: torch.Tensor,
    snr_linear: float,
    obs_snr_db: float,
    sigma_y2_like: float,
    num_steps: int | None,
    exp_key: str,
    cov_scale_mode: str,
    cov_beta_power: float | None,
    cov_grad_norm: str,
    cov_step_clip: float | None,
    cov_clip_mode: str,
    like_beta_power: float,
    like_snr_gate: bool,
    like_snr0_db: float,
    like_snr_delta_db: float,
    seed: int,
    device: torch.device,
    fft_pre: bool,
    mode: str,
    H_star_eval: torch.Tensor,
    H_star_in: torch.Tensor,
    sample_indices: Sequence[int],
) -> List[Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    likelihood_grad_fn = make_awgn_likelihood_grad(float(sigma_y2_like))

    sampler = DpsSampler(
        dm=dm,
        likelihood_grad_fn=likelihood_grad_fn,
        lambda_dps=float(cfg.dps_lambda),
        cov_lambda=float(cfg.cov_lambda),
        cov_scale_mode=str(cov_scale_mode),
        cov_beta_power=cov_beta_power,
        cov_grad_norm=str(cov_grad_norm),
        cov_step_clip=cov_step_clip,
        cov_clip_mode=str(cov_clip_mode),
        sigma_y2=float(sigma_y2_like),
        add_random=False,
        exp_key=str(exp_key),
        gamma=0.5,
        like_weight=1.0,
        lw_schedule="constant",
        lw_tau=1.0,
        lw_max=1.0,
        lw_end=0.0,
        lw_k=1.0,
        g_tau1=1.0,
        like_beta_power=float(like_beta_power),
        like_snr_gate=bool(like_snr_gate),
        like_snr0_db=float(like_snr0_db),
        like_snr_delta_db=float(like_snr_delta_db),
    )

    # Generate a single posterior sample for the fixed y.
    # SNR-matching enabled by passing snr=snr_linear (NOT None).
    H_hat_in = sampler.generate_posterior_sample(
        y_ri.to(device=device),
        cov=cov_ri.to(device=device),
        return_all_timesteps=False,
        num_steps=num_steps,
        snr=float(snr_linear),
        obs_snr_db=float(obs_snr_db),
        diagnostic_recorder=None,
        x_T=None,
    )

    # ------------------------------------------------------------
    # Metrics domain policy (user request):
    # "Don't consider FFT": compute ALL metrics in the same domain as final NMSE,
    # i.e., the eval/IFFT domain when fft_pre is enabled.
    # ------------------------------------------------------------
    H_hat_eval = maybe_ifft_post(H_hat_in, fft_pre=bool(fft_pre), mode=mode)
    y_eval = maybe_ifft_post(y_ri.to(device=device), fft_pre=bool(fft_pre), mode=mode)

    # Empirical noise variance in eval domain (per-sample), since FFT/IFFT might not be perfectly unitary.
    noise_eval = _to_complex(y_eval) - _to_complex(H_star_eval.to(device=device))
    sigma_y2_eval = torch.mean(torch.abs(noise_eval) ** 2, dim=(1, 2)).clamp(min=EPS)  # (B,)

    # Data consistency in eval domain
    res_eval = _to_complex(y_eval) - _to_complex(H_hat_eval)
    sq_eval = torch.abs(res_eval) ** 2
    ey_mse = torch.mean(sq_eval, dim=(1, 2))                   # (B,)
    ey_nll = torch.sum(sq_eval, dim=(1, 2)) / sigma_y2_eval     # (B,)

    # Gram consistency in eval domain (trace-normalized)
    G_ref = trace_normalize_gram(compute_gram_complex(_to_complex(H_star_eval.to(device=device))))
    G_hat = trace_normalize_gram(compute_gram_complex(_to_complex(H_hat_eval)))
    eG = frob_norm_complex(G_hat - G_ref)                       # (B,)

    # NMSE in eval domain (DMCE-style normalization)
    nmse = compute_nmse_dmce_style(H_hat_eval, H_star_eval.to(device=device))

    rows: List[Dict[str, float]] = []
    for bi, sidx in enumerate(sample_indices):
        rows.append(
            {
                "seed": float(seed),
                "sample_idx": float(sidx),
                "e_y_mse": float(ey_mse[bi].item()),
                "e_y_nll": float(ey_nll[bi].item()),
                "e_G": float(eG[bi].item()),
                "nmse": float(nmse[bi].item()),
            }
        )
    return rows


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows: List[Dict[str, float]], name: str) -> None:
    def _stat(k: str) -> str:
        v = np.array([r[k] for r in rows], dtype=float)
        return f"mean={v.mean():.4g}, std={v.std():.4g}, p10={np.percentile(v,10):.4g}, p50={np.percentile(v,50):.4g}, p90={np.percentile(v,90):.4g}"

    print(f"\n[{name}] n={len(rows)}")
    print(f"  e_y_mse: {_stat('e_y_mse')}")
    print(f"  e_y_nll: {_stat('e_y_nll')}")
    print(f"  e_G:     {_stat('e_G')}  (Gram in eval/IFFT domain; consistent with e_y and NMSE)")
    print(f"  nmse:    {_stat('nmse')}")


def plot_scatter(out_path: Path, data: Dict[str, List[Dict[str, float]]], *, y_key: str, y_label: str) -> None:
    plt.figure(figsize=(7.2, 5.7))
    ax = plt.gca()
    styles = {
        "likelihood_only": dict(marker="o", color="#17becf", label="Likelihood only"),
        "gram_only": dict(marker="s", color="#2ca02c", label="Gram only"),
        "gram_plus_likelihood": dict(marker="^", color="#d62728", label="Gram + Likelihood"),
    }
    for key, rows in data.items():
        x = np.array([r["e_y_mse"] for r in rows], dtype=float)
        y = np.array([r[y_key] for r in rows], dtype=float)
        st = styles.get(key, dict(marker="o", color="black", label=key))
        ax.scatter(x, y, s=28, alpha=0.85, edgecolors="none", **st)
    ax.set_xlabel(r"Data consistency $e_y$ (MSE)", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.legend(loc="best", fontsize=10, framealpha=0.95, fancybox=True, edgecolor="gray", frameon=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[saved] {out_path}")


def plot_box(out_path: Path, data: Dict[str, List[Dict[str, float]]], key: str, ylabel: str) -> None:
    plt.figure(figsize=(7.0, 5.2))
    ax = plt.gca()
    labels = ["Likelihood only", "Gram only", "Gram + Likelihood"]
    order = ["likelihood_only", "gram_only", "gram_plus_likelihood"]
    vals = [np.array([r[key] for r in data[k]], dtype=float) for k in order]
    ax.boxplot(vals, labels=labels, showfliers=False)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"[saved] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Conditional Gram consistency diagnostic (fixed y, multi-seed)")
    ap.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory (contains sim_params/ and train_models/)")
    ap.add_argument("--ch_type", type=str, default="quadriga_LOS", help="quadriga_LOS or 3gpp")
    ap.add_argument("--n_rx", type=int, default=64)
    ap.add_argument("--n_tx", type=int, default=16)
    ap.add_argument("--n_path", type=int, default=3)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--sample_idx", type=int, default=0, help="Which ground-truth sample index (within split) to fix")
    ap.add_argument("--n_samples_avg", type=int, default=1,
                    help="If >1, evaluate sample_idx..sample_idx+n_samples_avg-1 and report distribution across samples "
                         "(useful to compare against mean curves in plot_Quadriga_LOS.py).")
    ap.add_argument("--batch_samples", type=int, default=64,
                    help="How many ground-truth samples to process per GPU batch (speed knob).")
    ap.add_argument("--debug_gram_refs", action="store_true",
                    help="Print Gram-reference sanity checks (V1/V2) for the first processed batch.")
    ap.add_argument("--snr_db", type=float, default=5.0, help="Observation SNR in dB (high SNR recommended)")
    ap.add_argument("--num_steps", type=int, default=0, help="If >0, override sampler steps; if 0, use SNR-matched steps")
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default=str(Path("results") / "useful_results"))
    ap.add_argument("--fft_pre", action="store_true", help="Mirror evaluation: generate y in FFT-pre domain and IFFT estimates for NMSE.")
    ap.add_argument("--mode", type=str, default="2D", choices=["1D", "2D"], help="FFT mode (match evaluation scripts).")

    # Guidance hyperparams
    ap.add_argument("--exp_key", type=str, default="H", help="Likelihood experiment key (H recommended)")
    ap.add_argument("--dps_lambda", type=float, default=0.1)
    ap.add_argument("--cov_lambda", type=float, default=0.01)
    ap.add_argument("--method", type=str, default="dps_cov_oracle", choices=["dps", "dps_cov_oracle", "dps_cov_est"])
    ap.add_argument("--cov_scale_mode", type=str, default="sqrt_beta_t")
    ap.add_argument("--cov_beta_power", type=float, default=float("nan"), help="If set (not NaN), overrides cov_scale_mode with zeta=beta^p")
    ap.add_argument("--cov_grad_norm", type=str, default="none")
    ap.add_argument("--cov_step_clip", type=float, default=2.0)
    ap.add_argument("--cov_clip_mode", type=str, default="norm", choices=["auto", "elementwise", "norm"])
    ap.add_argument("--n_time_samples", type=int, default=2000, help="N_d for dps_cov_est time-averaged covariance.")
    ap.add_argument("--modulation", type=str, default="qpsk", choices=["bpsk", "qpsk"])
    ap.add_argument("--like_beta_power", type=float, default=1.0)
    ap.add_argument("--like_snr_gate", action="store_true")
    ap.add_argument("--like_snr0_db", type=float, default=-10.5)
    ap.add_argument("--like_snr_delta_db", type=float, default=2.0)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_dir = Path(args.model_dir)
    dm = load_diffusion_model(model_dir, device=str(device))
    dm.to(device)

    # Generate fixed observation SNR params
    snr_linear = float(10 ** (args.snr_db / 10.0))

    # Sigma_y2 aligned with AWGN in evaluation pipeline: (noise_mult^2) / rho
    sigma_y2_like = float((float(dm.noise_multiplier) ** 2) / float(snr_linear))

    # Choose steps override
    num_steps = None if args.num_steps <= 0 else int(args.num_steps)

    cov_beta_power = None if (isinstance(args.cov_beta_power, float) and np.isnan(args.cov_beta_power)) else float(args.cov_beta_power)

    # Three modes
    configs = {
        "likelihood_only": RunConfig("likelihood_only", dps_lambda=float(args.dps_lambda), cov_lambda=0.0),
        "gram_only": RunConfig("gram_only", dps_lambda=0.0, cov_lambda=float(args.cov_lambda)),
        "gram_plus_likelihood": RunConfig("gram_plus_likelihood", dps_lambda=float(args.dps_lambda), cov_lambda=float(args.cov_lambda)),
    }

    all_rows: Dict[str, List[Dict[str, float]]] = {k: [] for k in configs.keys()}

    print("\n[Setup]")
    print(f"  device={device}")
    print(f"  ch_type={args.ch_type}, split={args.split}, Nr={args.n_rx}, Nt={args.n_tx}, "
          f"sample_idx={args.sample_idx}, n_samples_avg={args.n_samples_avg}")
    print(f"  snr_db={args.snr_db:.1f} (rho={snr_linear:.4g}), sigma_y2_like={sigma_y2_like:.4g}")
    print(f"  exp_key={args.exp_key}, n_seeds={args.n_seeds}, seed0={args.seed0}")
    print(f"  method={args.method}, fft_pre={bool(args.fft_pre)}, mode={args.mode}")
    print(f"  cov: scale_mode={args.cov_scale_mode}, beta_power={cov_beta_power}, grad_norm={args.cov_grad_norm}, clip={args.cov_clip_mode}, step_clip={args.cov_step_clip}")
    print(f"  like: dps_lambda={args.dps_lambda}, like_beta_power={args.like_beta_power}, like_snr_gate={args.like_snr_gate}")

    # Fast path: load requested slice once, then process in GPU batches.
    H_split_np, ch_tag = prepare_split_numpy(
        ch_type=args.ch_type,
        n_rx=args.n_rx,
        n_tx=args.n_tx,
        n_path=args.n_path,
        n_train=100_000,
        n_val=10_000,
        n_test=10_000,
        split=args.split,
        start_idx=int(args.sample_idx),
        n_samples=int(args.n_samples_avg),
    )

    total = int(args.n_samples_avg)
    bs = max(1, int(args.batch_samples))
    for start in range(0, total, bs):
        end = min(total, start + bs)
        idxs = list(range(int(args.sample_idx) + start, int(args.sample_idx) + end))

        H_star_ri = complex_numpy_to_ri_torch(H_split_np[start:end]).to(device=device)  # (B,2,Nr,Nt)
        H_star_in = maybe_fft_pre(H_star_ri, fft_pre=bool(args.fft_pre), mode=args.mode).to(device=device)
        H_star_eval = H_star_ri

        y_ri = functional.awgn(H_star_in, snr_linear, multiplier=dm.noise_multiplier).to(device=device)

        if args.method in ("dps", "dps_cov_oracle"):
            cov_ri = generate_cov_batch(H_star_in).to(device=device)
        else:
            cov_ri = estimate_cov_time_averaged_batch(
                H_star_in,
                snr_db=float(args.snr_db),
                n_time_samples=int(args.n_time_samples),
                modulation=str(args.modulation),
            ).to(device=device)

        # ------------------------------------------------------------
        # Debug V1/V2: Is the Gram reference used for guidance the same as the Gram used for e_G?
        # Only print once (first processed batch) to avoid spam.
        # ------------------------------------------------------------
        if args.debug_gram_refs and start == 0:
            dbg = debug_gram_reference_consistency(
                H_star_in=H_star_in,
                H_star_eval=H_star_eval,
                cov_ri=cov_ri,
            )
            print("\n[Debug] Gram reference consistency checks (trace-normalized Frobenius distances)")
            print("  V1: ||G_ref,guide - G*_in||_F  =", f"{dbg['d_Gguide_vs_Gstar_in_trace']:.4f}")
            print("  V1: ||G_ref,guide - G*_eval||_F =", f"{dbg['d_Gguide_vs_Gstar_eval_trace']:.4f}")
            print("  V2: ||G*_in - G*_eval||_F       =", f"{dbg['d_Gstar_in_vs_eval_trace']:.4f}")

        for si in range(args.n_seeds):
            seed = int(args.seed0 + si)
            for key, cfg in configs.items():
                rows = run_mode(
                    cfg=cfg,
                    dm=dm,
                    H_star_ri=H_star_ri,
                    y_ri=y_ri,
                    cov_ri=cov_ri,
                    snr_linear=snr_linear,
                    obs_snr_db=float(args.snr_db),
                    sigma_y2_like=sigma_y2_like,
                    num_steps=num_steps,
                    exp_key=args.exp_key,
                    cov_scale_mode=args.cov_scale_mode,
                    cov_beta_power=cov_beta_power,
                    cov_grad_norm=args.cov_grad_norm,
                    cov_step_clip=float(args.cov_step_clip) if args.cov_step_clip is not None else None,
                    cov_clip_mode=args.cov_clip_mode,
                    like_beta_power=float(args.like_beta_power),
                    like_snr_gate=bool(args.like_snr_gate),
                    like_snr0_db=float(args.like_snr0_db),
                    like_snr_delta_db=float(args.like_snr_delta_db),
                    seed=seed,
                    device=device,
                    fft_pre=bool(args.fft_pre),
                    mode=str(args.mode),
                    H_star_eval=H_star_eval,
                    H_star_in=H_star_in,
                    sample_indices=idxs,
                )
                all_rows[key].extend(rows)

    # Print summaries
    for key, rows in all_rows.items():
        summarize(rows, key)

    # Save raw metrics
    out_dir = Path(args.out_dir)
    tag = f"{ch_tag}_snr{args.snr_db:.1f}_split{args.split}_idx{args.sample_idx}_n{args.n_samples_avg}_exp{args.exp_key}"
    npz_path = out_dir / f"cond_diag_{tag}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        **{
            f"{k}_seed": np.array([r["seed"] for r in v], dtype=float)
            for k, v in all_rows.items()
        },
        **{
            f"{k}_sample_idx": np.array([r["sample_idx"] for r in v], dtype=float)
            for k, v in all_rows.items()
        },
        **{
            f"{k}_e_y_mse": np.array([r["e_y_mse"] for r in v], dtype=float)
            for k, v in all_rows.items()
        },
        **{
            f"{k}_e_y_nll": np.array([r["e_y_nll"] for r in v], dtype=float)
            for k, v in all_rows.items()
        },
        **{
            f"{k}_e_G": np.array([r["e_G"] for r in v], dtype=float)
            for k, v in all_rows.items()
        },
        **{
            f"{k}_nmse": np.array([r["nmse"] for r in v], dtype=float)
            for k, v in all_rows.items()
        },
    )
    print(f"\n[saved] {npz_path}")

    # CSVs per mode
    for key, rows in all_rows.items():
        save_csv(out_dir / f"cond_diag_{tag}_{key}.csv", rows)

    # Plots
    plot_scatter(
        out_dir / f"cond_diag_scatter_e_G_{tag}.png",
        all_rows,
        y_key="e_G",
        y_label=r"Gram consistency $e_G$ (eval/IFFT domain)",
    )
    plot_box(out_dir / f"cond_diag_box_e_y_{tag}.png", all_rows, key="e_y_mse", ylabel=r"Data consistency $e_y$ (MSE)")
    plot_box(out_dir / f"cond_diag_box_e_G_{tag}.png", all_rows, key="e_G", ylabel=r"Gram consistency $e_G$ (eval/IFFT domain)")
    plot_box(out_dir / f"cond_diag_box_nmse_{tag}.png", all_rows, key="nmse", ylabel="NMSE vs $H_\\star$")

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()


