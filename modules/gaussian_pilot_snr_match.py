"""
Gaussian spatial pilots: LS channel, timestep and init aligned with the orthonormal pilot schedule.

X_p = sqrt(γ) I_rect + sqrt(1-γ) G (γ=0 reduces to pure G in load_and_eval_dm_dps).

Y_p = H X_p + N with N ~ CN(0, sigma^2 I) (per complex entry, matching the simulator).

H_hat_LS = Y_p X_p^H (X_p X_p^H)^{-1}.

**Timestep SNR (matches orthonormal when C = X_p X_p^H = I):**

    SNR_eff = rho_linear * N_t / tr(C^{-1})

with C the regularized Gram matrix (same jitter as in LS). If C = I_{N_t}, then tr(C^{-1}) = N_t
and SNR_eff = rho_linear, i.e. the same linear SNR used in the orthogonal branch for
``argmin_t |SNR_DM(t) - rho|``.

This is algebraically equivalent to N_t / (sigma^2 tr(C^{-1})) once sigma^2 and rho are linked
by the simulation noise model so that N_t/(sigma^2 tr) collapses to rho at C = I.

**Init:** energy normalization from the scalar LS effective noise (same as the pre-rho-scaling code),

    eta_eff^2 = sigma^2 * tr(C^{-1}) / N_t,
    x_init = FFT(H_LS) / sqrt(1 + eta_eff^2),

so expected power is scaled down consistently with the estimated post-LS error variance.

**SNR match mode ``worst`` (optional):** uses the smallest eigenvalue of the *unjittered* Gram
`C = X_p X_p^{\\mathsf H}`: `\\sigma_{\\mathrm{eff}}^2 = \\sigma^2 / \\lambda_{\\min}(C)`,
`\\mathrm{SNR}_{\\mathrm{eff}} = \\rho\\,\\lambda_{\\min}(C)` (same linear SNR as the trace mode when
`C = I`). `C` and `\\lambda_{\\min}` are computed once per ``build_gaussian_snr_match`` call and
returned for reuse.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

import modules.pilot_matrix as pm
import modules.utils as ut


def _sigma2_complex_pilot_entry(rho: float, noise_multiplier: float) -> float:
    """
    Match load_and_eval_dm_dps Gaussian pilot: scale = mult/sqrt(rho), nr, ni ~ N(0, scale^2).
    Per complex entry E|n|^2 = 2 * scale^2 = 2 * mult^2 / rho.
    """
    m = float(noise_multiplier)
    r = float(rho)
    return 2.0 * (m * m) / r


def least_squares_channel_batch(
    Y_c: torch.Tensor,
    X_p: torch.Tensor,
    jitter: float = 1e-6,
    gram_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    H_ls = Y_p X_p^H (X_p X_p^H)^{-1}.

    Y_c: (B, N_r, N_p) complex
    X_p: (N_t, N_p) complex
    gram_inv: optional precomputed (X_p X_p^H + jitter I)^{-1} to avoid duplicate Gram work.
    Returns H_ls: (B, N_r, N_t) complex
    """
    if gram_inv is None:
        gram = X_p @ X_p.conj().transpose(-1, -2)
        nt = gram.shape[0]
        eye = torch.eye(nt, dtype=gram.dtype, device=gram.device)
        gram = gram + jitter * eye
        gram_inv = torch.linalg.inv(gram)
    XpH = X_p.conj().transpose(-1, -2)
    return torch.matmul(torch.matmul(Y_c, XpH), gram_inv)


def eta_eff_sq_from_gram(
    gram_inv: torch.Tensor,
    sigma2: float,
    n_t: int,
) -> torch.Tensor:
    """eta_eff^2 = sigma^2 * (1/N_t) * tr(gram^{-1}), scalar tensor (diagnostic / legacy)."""
    tr = torch.trace(gram_inv).real
    return (sigma2 / float(n_t)) * tr


def match_timestep_snr(
    snr_eff: Union[float, torch.Tensor],
    dm_snrs: torch.Tensor,
) -> int:
    """t* = argmin |SNR_eff - SNR_DM(t)|; SNR_DM = alpha_bar/(1-alpha_bar) (isotropic DDPM)."""
    s = snr_eff if isinstance(snr_eff, torch.Tensor) else torch.tensor(snr_eff, dtype=dm_snrs.dtype, device=dm_snrs.device)
    return int(torch.abs(dm_snrs - s).argmin().item())


def spatial_ls_to_angular(
    H_ls_c: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """H_ls_c: (B, N_r, N_t) complex -> (B, 2, N_r, N_t) real/imag angular domain."""
    h_sp = torch.stack((H_ls_c.real, H_ls_c.imag), dim=1)
    return ut.complex_1d_fft(h_sp, ifft=False, mode=mode)


def build_gaussian_snr_match(
    *,
    Y_p_c: torch.Tensor,
    X_p: torch.Tensor,
    rho_linear: float,
    noise_multiplier: float,
    dm_snrs: torch.Tensor,
    spatial_fft_mode: str,
    eta_mode: str = "per_sample",
    dataset_avg_trace_over_nt: Optional[float] = None,
    snr_match_mode: str = "trace",
    dataset_avg_inv_lambda_min: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns dict with keys: eta_eff_sq, snr_eff, t_start, x_init_ang, gram_inv, sigma2,
    plus ``gram_c`` (unjittered ``C = X_p X_p^H``) and, for ``snr_match_mode='worst'``,
    ``lambda_min``, ``sigma_eff2``.

    snr_match_mode ``trace`` (default): snr_eff = rho_linear * N_t / tr(C^{-1}) with jittered C;
    eta from tr(C^{-1})/N_t (or dataset_avg_trace_over_nt).

    snr_match_mode ``worst``: C unjittered, lambda_min = min eig(C), sigma_eff2 = sigma2/lambda_min,
    snr_eff = rho_linear * lambda_min (equivalently rho * sigma2 / sigma_eff2); same t* matching.
    eta per_sample uses sigma_eff2; dataset_avg uses dataset_avg_inv_lambda_min = E[1/lambda_min].

    eta_mode dataset_avg only affects eta_eff_sq (and thus init); t* still uses this draw's Gram
    statistics (trace or lambda_min).
    """
    n_t = X_p.shape[0]
    gram_c = X_p @ X_p.conj().transpose(-1, -2)
    eye = torch.eye(n_t, dtype=gram_c.dtype, device=gram_c.device)
    gram = gram_c + 1e-6 * eye
    gram_inv = torch.linalg.inv(gram)

    tr_inv = torch.trace(gram_inv).real.clamp(min=1e-12)
    tr_over_nt_tensor = tr_inv / float(n_t)

    sigma2 = _sigma2_complex_pilot_entry(rho_linear, noise_multiplier)
    mode = str(snr_match_mode).strip().lower()
    rho_f = float(rho_linear)

    if mode == "trace":
        if eta_mode == "dataset_avg":
            if dataset_avg_trace_over_nt is None:
                raise ValueError("dataset_avg_trace_over_nt required for eta_mode='dataset_avg'")
            eta_eff_sq = torch.tensor(
                sigma2 * float(dataset_avg_trace_over_nt),
                dtype=Y_p_c.dtype,
                device=Y_p_c.device,
            )
        else:
            eta_eff_sq = sigma2 * tr_over_nt_tensor
        snr_eff = rho_f * float(n_t) / float(tr_inv.item())
        lambda_min_t: Optional[torch.Tensor] = None
        sigma_eff2_val: Optional[float] = None
    elif mode == "worst":
        evals = torch.linalg.eigvalsh(gram_c)
        lambda_min_t = evals.min().real.clamp(min=1e-12)
        sigma_eff2_t = sigma2 / lambda_min_t
        sigma_eff2_val = float(sigma_eff2_t.item())
        snr_eff = rho_f * float(lambda_min_t.item())
        if eta_mode == "dataset_avg":
            if dataset_avg_inv_lambda_min is None:
                raise ValueError(
                    "dataset_avg_inv_lambda_min required for eta_mode='dataset_avg' "
                    "when snr_match_mode='worst'"
                )
            eta_eff_sq = torch.tensor(
                sigma2 * float(dataset_avg_inv_lambda_min),
                dtype=Y_p_c.dtype,
                device=Y_p_c.device,
            )
        else:
            eta_eff_sq = sigma_eff2_t.to(device=Y_p_c.device)
    else:
        raise ValueError("snr_match_mode must be 'trace' or 'worst'")

    t_start = match_timestep_snr(snr_eff, dm_snrs.to(device=Y_p_c.device))

    H_ls = least_squares_channel_batch(Y_p_c, X_p, gram_inv=gram_inv)
    h_ang = spatial_ls_to_angular(H_ls, spatial_fft_mode)
    scale = torch.sqrt(1.0 + eta_eff_sq.to(dtype=h_ang.dtype, device=h_ang.device))
    x_init = h_ang / scale.view(1, 1, 1, 1)

    out: Dict[str, Any] = {
        "eta_eff_sq": eta_eff_sq,
        "snr_eff": snr_eff,
        "t_start": t_start,
        "x_init_ang": x_init,
        "gram_inv": gram_inv,
        "sigma2": sigma2,
        "gram_c": gram_c.detach(),
        "snr_match_mode": mode,
    }
    if mode == "worst":
        out["lambda_min"] = lambda_min_t.detach()
        out["sigma_eff2"] = sigma_eff2_val
    return out


@torch.no_grad()
def monte_carlo_mean_trace_inv_over_nt(
    n_t: int,
    n_p: int,
    n_draws: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
    spatial_pilot_gamma: float = 0.0,
) -> float:
    """
    E[ tr((X_p X_p^H)^{-1}) / N_t ] over X_p = sqrt(g)*I_rect + sqrt(1-g)*G (same as pilots).
    Used for eta_mode=dataset_avg.
    """
    acc = 0.0
    for _ in range(n_draws):
        X = pm.draw_xp_sqrt_gamma_identity_gaussian_torch(
            n_t,
            n_p,
            spatial_pilot_gamma,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        gram = X @ X.conj().transpose(-1, -2)
        eye = torch.eye(n_t, dtype=gram.dtype, device=device)
        gram = gram + 1e-6 * eye
        ginv = torch.linalg.inv(gram)
        acc += (torch.trace(ginv).real / float(n_t)).item()
    return acc / float(n_draws)


@torch.no_grad()
def monte_carlo_mean_inv_lambda_min(
    n_t: int,
    n_p: int,
    n_draws: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
    spatial_pilot_gamma: float = 0.0,
) -> float:
    """
    E[ 1 / lambda_min(X_p X_p^H) ] over X_p = sqrt(g)*I_rect + sqrt(1-g)*G (same as pilots).
    Used for eta_mode=dataset_avg when snr_match_mode=worst.
    """
    acc = 0.0
    for _ in range(n_draws):
        X = pm.draw_xp_sqrt_gamma_identity_gaussian_torch(
            n_t,
            n_p,
            spatial_pilot_gamma,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        C = X @ X.conj().transpose(-1, -2)
        ev = torch.linalg.eigvalsh(C)
        lam = float(ev.min().real.clamp(min=1e-12).item())
        acc += 1.0 / lam
    return acc / float(n_draws)
