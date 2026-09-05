"""
DPS (Diffusion Posterior Sampling) Sampler for Channel Estimation.

This module implements DPS on top of an unconditional DiffusionModel, allowing
posterior sampling without retraining.

Key Design Decisions:
1. Gradient correction is applied AFTER the prior reverse step (not before)
2. Deterministic DDIM-style steps (add_random=False) are recommended
3. Step size scales with beta_t (forward process noise variance)
4. Optional clipping of the likelihood correction (per-sample L2 norm or elementwise) and cov correction

Theory:
For observation model y = x + n, n ~ N(0, sigma_y^2 I):
- Likelihood gradient: ∇_x log p(y|x) = (y - x) / sigma_y^2
- DPS update: x_prev = x_prior + lambda_dps * beta_t * grad(x_prior, y)

Domain (with `load_and_eval_dm_dps.py` defaults):
- **Angular diffusion:** the unconditional DM is trained on the **angular-domain** channel
  ``\\tilde{H}`` (same unitary FFT as the channel simulator). Reverse diffusion, Gram guidance,
  and likelihood guidance all update ``\\tilde{H}_t`` in that domain; the final estimate is mapped
  back to spatial with the inverse FFT if needed.
- **Paper / Algorithm 1 view (orthonormal pilots):** pilot observations are reduced to an
  **angular effective observation** ``\\tilde{Y}`` by composing DFTs with pilot-side decorrelation,
  schematically ``\\tilde{Y} \\leftarrow \\Phi_{N_R} Y_p X_p^H (X_p X_p^H)^{-1} \\Phi_{N_T}^H`` (exact
  layout matches your notation). Multiuser **block-orthonormal** ``X_p`` is the same idea with
  block structure: still formulate ``\\tilde{Y}``, ``\\tilde{H}``, and (Gram) ``\\tilde{R}`` in the
  angular domain, then run diffusion there. Spatial AWGN on the grid maps to complex Gaussian noise
  in angular coordinates under the usual circulant / 2-D DFT model.
- **Implementation vs notation:** this repo may build likelihood directions by applying ``X_p^H`` on
  **spatial** ``Y_p`` and ``\\hat{H}`` and then FFT-ing, or by precomputing ``\\tilde{Y}`` once and
  differencing against ``\\mathcal{T}(\\tilde{H}_t)`` in angular space. Those are the **same linear
  map** written in two orders; it is not a different physical model.
- **Why FFT for non-orthonormal pilots:** the goal is **representation alignment** with the
  orthonormal (angular) pipeline and the DM trained on ``\\tilde{H}``, **not** to claim that the
  pilot noise becomes white in angular coordinates. After FFT, effective noise covariance is
  generally **not** proportional to identity; treating likelihood or timestep choice as if it were
  is only a convenience / approximation.
- **Pilot likelihood domain (Gaussian / nonorthogonal pilots):** default ``pilot_likelihood_domain='spatial'``
  keeps the existing spatial residual then FFT. Optional ``'angular_ls'`` forms ``Y' = Y_p X_p^H C^{-1}``,
  FFTs to ``Y_tilde``, and uses ``(Y_tilde - \\hat{\\tilde{H}}_0)\\,(F_{tx} C F_{tx}^{\\mathsf H})`` with
  ``C = X_p X_p^{\\mathsf H}`` (see ``angular_ls_mahalanobis_likelihood_grad_angular`` and
  ``tx_gram_spatial_to_angular``).
- **``SNR_eff`` / ``t*`` matching (``gaussian_pilot_snr_match``):** maps a **scalar** summary of LS
  effective noise (via ``\\mathrm{tr}(C^{-1})``) onto the **isotropic** DDPM schedule ``\\bar{\\alpha}_t
  / (1-\\bar{\\alpha}_t)``. That is a **heuristic** for choosing ``t^*`` and init scale; it does **not**
  rigorously match the true posterior or the colored noise after pilot compression and FFT.

**Mahalanobis / non-orthonormal pilots.** With ``C = X_p X_p^H``, the spatial equivalent model
``Y' = H + N'`` has ``\\mathrm{Cov}(N') = \\sigma^2 C^{-1}``. The correct score w.r.t. ``H`` is
proportional to ``(Y_p - H X_p)X_p^H`` (equivalently ``(Y'-H)C``). In angular coordinates that
appears as the **composed** operator, not a plain entrywise ``\\tilde{Y} - \\tilde{H}``, unless
``C = I`` (orthonormal pilots). When ``X_p \\approx I`` (square), ``(Y_p - H X_p)X_p^H \\approx Y_p - H``
and ``FFT(Y_p) - x0_hat`` matches the orthonormal-pilot likelihood shortcut.
"""

from typing import Callable, Optional, Literal
import math
import warnings

import torch
from torch import Tensor
from DMCE import utils as dm_utils
import modules.utils as ut
from modules import pilot_matrix as _pm_pilot


# -------------------------- Likelihood gradient -------------------------- #

def make_awgn_likelihood_grad(sigma_y2: float) -> Callable[[Tensor, Tensor, int], Tensor]:
    """
    AWGN likelihood gradient for model: y = x + n,  n ~ N(0, sigma_y2 I).

    Then
        ∇_x log p(y | x) = (y - x) / sigma_y2.
    """
    sigma_y2 = float(sigma_y2)

    def likelihood_grad_fn(x_ref: Tensor, y: Tensor, t: int) -> Tensor:
        # t is unused for simple AWGN, but kept for API compatibility
        _ = t
        # Standardize (y - x_ref) by its L2 norm (unit vector)
        diff = y - x_ref
        # Compute L2 norm for each sample in the batch
        norm_dim = tuple(range(1, diff.ndim))  # All dims except batch dim
        norm = torch.linalg.vector_norm(diff, dim=norm_dim, keepdim=True)
        # Add small epsilon to avoid division by zero
        return diff / (sigma_y2)

    return likelihood_grad_fn


def _tweedie_x0_hat_pred_noise(x_t_ri: Tensor, dm, t: int) -> Tensor:
    """DDPM/Tweedie x0_hat(H_t) from pred_noise objective (same construction as exp H)."""
    if dm.objective != "pred_noise":
        raise NotImplementedError(
            "Gaussian-pilot likelihood requires DiffusionModel objective='pred_noise'."
        )
    b, *_ = x_t_ri.shape
    batched_times = torch.full((b,), t, device=dm.device, dtype=torch.long)
    eps_hat = dm.model(x_t_ri, batched_times)
    a = torch.clamp(dm.alphas_cumprod[t].to(x_t_ri.device), min=1e-12)
    shape_ones = (1,) * (x_t_ri.ndim - 1)
    a_view = a.view(1, *shape_ones)
    one_minus_a = torch.clamp(1.0 - a_view, min=1e-12)
    score_prior = -eps_hat / torch.sqrt(one_minus_a)
    x0_hat = (x_t_ri + one_minus_a * score_prior) / torch.sqrt(a_view + 1e-12)
    return x0_hat


def nonorthogonal_pilot_likelihood_grad_angular(
    x_t_ri: Tensor,
    gp_ctx: dict,
    dm,
    t: int,
    sigma_y2: float,
) -> Tensor:
    """
    DPS likelihood direction from the equivalent model ``Y' = Y_p X_p^H C^{-1} = H + N'`` with
    ``C = X_p X_p^H``: gradient w.r.t. ``H`` (spatial) is proportional to ``(Y' - H_0) C``.
    Map to angular with the same FFT as ``H`` (entrywise linearity in the channel matrix).

    Divides by ``sigma_y2`` so this matches orthogonal Exp H, which uses ``(y - x0_hat) / sigma_y2``.
    """
    fft_mode = str(gp_ctx["fft_mode"])
    x0_hat = _tweedie_x0_hat_pred_noise(x_t_ri, dm, t)
    H0_sp = ut.complex_1d_fft(x0_hat, ifft=True, mode=fft_mode, _4d_array=False)
    if H0_sp.dim() != 4 or H0_sp.size(1) != 2:
        raise ValueError(f"Expected H0_sp (B,2,N_R,N_T), got {tuple(H0_sp.shape)}")
    H0c = torch.complex(H0_sp[:, 0], H0_sp[:, 1])
    Y_prime = gp_ctx["Y_prime_c"].to(device=H0c.device, dtype=H0c.dtype)
    gram = gp_ctx["gram_c"].to(device=H0c.device, dtype=H0c.dtype)
    if Y_prime.shape != H0c.shape:
        raise ValueError(f"Y_prime_c {tuple(Y_prime.shape)} vs H0 {tuple(H0c.shape)}")
    B = Y_prime.shape[0]
    gram_b = gram.unsqueeze(0).expand(B, -1, -1)
    G_sp = torch.bmm(Y_prime - H0c, gram_b)
    G_ri = torch.stack((G_sp.real, G_sp.imag), dim=1)
    out = ut.complex_1d_fft(G_ri, ifft=False, mode=fft_mode, _4d_array=False)
    return out / float(sigma_y2)


def gaussian_pilot_likelihood_grad_angular(
    x_t_ri: Tensor,
    gp_ctx: dict,
    dm,
    t: int,
    sigma_y2: float,
) -> Tensor:
    """
    DPS-style likelihood direction for ``Y_p = H X_p + N`` mapped to angular coordinates.

    The score w.r.t. ``H`` (spatial) is proportional to ``(Y_p - H X_p)X_p^H``, i.e.
    ``(Y' - H)C`` with ``Y' = Y_p X_p^H C^{-1}`` and ``C = X_p X_p^H`` (Mahalanobis form for
    ``Y' = H + N'``). We return ``FFT`` of that matrix (same convention as the DM), which is the
    direction used in angular-state DPS.

    If ``gp_ctx['gaussian_likelihood_angular']`` (``N_T = N_P`` and ``X_p \\approx I``): then
    ``(Y_p - H_0 X_p)X_p^H \\approx Y_p - H_0`` and ``FFT(Y_p - H_0) = Y_p^{ang} - x0_hat``;
    use that fast path. For general ``X_p`` (including ``C = I`` but ``X_p \\neq I``), the full
    spatial multiply is required; do **not** use ``Y_p^{ang} - x0_hat`` alone.

    Output is scaled by ``1/sigma_y2`` like ``make_awgn_likelihood_grad``.
    """
    fft_mode = str(gp_ctx["fft_mode"])
    X_p = gp_ctx["X_p"]
    x0_hat = _tweedie_x0_hat_pred_noise(x_t_ri, dm, t)
    s2 = float(sigma_y2)

    if gp_ctx.get("gaussian_likelihood_angular"):
        Y_p_ang = gp_ctx["Y_p_ang"].to(device=x0_hat.device, dtype=x0_hat.dtype)
        return (Y_p_ang - x0_hat) / s2

    Y_p_ri = gp_ctx["Y_p"]
    H0_sp = ut.complex_1d_fft(x0_hat, ifft=True, mode=fft_mode, _4d_array=False)
    if H0_sp.dim() != 4 or H0_sp.size(1) != 2:
        raise ValueError(f"Expected H0_sp (B,2,N_R,N_T), got {tuple(H0_sp.shape)}")
    Y_sp = torch.complex(Y_p_ri[:, 0], Y_p_ri[:, 1])
    if Y_sp.dim() != 3:
        raise ValueError(f"Expected Y_p_ri (B,2,N_R,N_p), got {tuple(Y_p_ri.shape)}")
    Xp = X_p.to(device=Y_sp.device, dtype=Y_sp.dtype)
    if Xp.dim() != 2:
        raise ValueError("X_p must have shape (N_T, N_P)")
    H0c = torch.complex(H0_sp[:, 0], H0_sp[:, 1])
    B, n_r, n_t = H0c.shape
    if Xp.shape[0] != n_t:
        raise ValueError(f"X_p rows {Xp.shape[0]} must equal N_T={n_t} from channel")
    Xb = Xp.unsqueeze(0).expand(B, -1, -1)
    residual = Y_sp - torch.bmm(H0c, Xb)
    G_sp = torch.bmm(residual, Xb.conj().transpose(-1, -2))
    G_ri = torch.stack((G_sp.real, G_sp.imag), dim=1)
    out = ut.complex_1d_fft(G_ri, ifft=False, mode=fft_mode, _4d_array=False)
    return out / s2


def angular_ls_mahalanobis_likelihood_grad_angular(
    x_t_ri: Tensor,
    gp_ctx: dict,
    dm,
    t: int,
    sigma_y2: float,
) -> Tensor:
    """
    Angular-domain LS-equivalent likelihood direction (Mahalanobis-style approximation).

    Spatial model ``Y' = Y_p X_p^H (X_p X_p^H)^{-1} = H + N'`` with ``C = X_p X_p^H``.
    After ``Y_tilde = FFT(Y')`` and ``H0_tilde = FFT(H0)`` (same ``complex_1d_fft`` as the DM),
    use

        grad_like = (Y_tilde - H0_tilde) @ C_prec_tilde,

    with ``C_prec_tilde = F_tx C F_tx^H`` from ``tx_gram_spatial_to_angular(C)`` (precision-side
    on Tx angular coordinates; **not** ``FFT(C^{-1})``).

    Returns real/imag stacked tensor matching ``x_t_ri``. Scaled by ``1/sigma_y2`` like other
    pilot branches for consistent outer DPS weighting.
    """
    x0_hat = _tweedie_x0_hat_pred_noise(x_t_ri, dm, t)
    H0_tilde = torch.complex(x0_hat[:, 0], x0_hat[:, 1])
    Y_tilde = gp_ctx["Y_tilde_c"].to(device=H0_tilde.device, dtype=H0_tilde.dtype)
    Cprec = gp_ctx["Cprec_tilde_c"].to(device=H0_tilde.device, dtype=H0_tilde.dtype)
    if Y_tilde.shape != H0_tilde.shape:
        raise ValueError(f"Y_tilde {tuple(Y_tilde.shape)} vs H0_tilde {tuple(H0_tilde.shape)}")
    B, n_r, n_t = H0_tilde.shape
    if Cprec.dim() != 2 or Cprec.shape != (n_t, n_t):
        raise ValueError(f"Cprec_tilde_c expected ({n_t}, {n_t}), got {tuple(Cprec.shape)}")
    residual = Y_tilde - H0_tilde
    Cb = Cprec.unsqueeze(0).expand(B, -1, -1)
    G = torch.bmm(residual, Cb)
    G_ri = torch.stack((G.real, G.imag), dim=1)
    return G_ri / float(sigma_y2)


def compute_cov_grad(x_ref, cov, sigma_y2):
    """
    Gradient of f(X) = -||X X^H - cov||_F^2 w.r.t. x_ref.

    x_ref: real tensor of shape (batch, 2, n_meas, n_input) where channel 0 is
        the real part and channel 1 is the imaginary part of X.
    cov: complex covariance stored as real/imag channels with shape
        (batch, 2, n_meas, n_meas); channel 0 is Re(cov), channel 1 is Im(cov).
        Batch dimension may be 1 to broadcast across x_ref.

    Returns: gradient with the same shape as x_ref.
    """
    if x_ref.dim() != 4 or x_ref.size(1) != 2:
        raise ValueError("x_ref must have shape (batch, 2, n_meas, n_input)")
    if cov.dim() != 4 or cov.size(1) != 2:
        raise ValueError("cov must have shape (batch, 2, n_meas, n_meas)")

    # Build complex matrix X (batch, n_meas, n_input)
    x_comp = torch.complex(x_ref[:, 0], x_ref[:, 1])

    # Convert covariance to complex and handle batch broadcasting
    cov_comp = torch.complex(cov[:, 0], cov[:, 1]).to(x_comp.device)
    cov_comp = cov_comp.type_as(x_comp)
    if cov_comp.size(0) == 1 and x_comp.size(0) > 1:
        cov_comp = cov_comp.expand(x_comp.size(0), -1, -1)
    if cov_comp.size(0) != x_comp.size(0):
        raise ValueError("Batch size of cov must match x_ref or be 1 for broadcast")

    # Current covariance: X X^H
    cov_current = x_comp @ x_comp.conj().transpose(-1, -2)

    # Gradient of -||X X^H - cov||_F^2 is 4 (cov - X X^H) X
    error = cov_comp - cov_current
    grad_comp = 4.0 * (error @ x_comp) 

    # Return real/imag parts stacked to match x_ref layout
    return torch.stack((grad_comp.real, grad_comp.imag), dim=1)


def compute_tx_cov_grad(x_ref: "Tensor", tau: float = 0.3) -> "Tensor":
    """
    Per-sample Tx Gram off-diagonal regularization with a soft-threshold (dead-zone).

    Motivation
    ----------
    True individual H^H H matrices are NOT perfectly diagonal; they have natural 
    off-diagonal correlation coefficients ranging roughly from 0.05 to 0.3.
    A strict L2 penalty drives these natural correlations to 0, causing a bias.
    This function computes the normalized correlation coefficient matrix and ONLY
    penalizes off-diagonal entries that exceed the threshold `tau`.

    Derivation
    ----------
    Let G = X^H X. The diagonal is D = diag(G).
    The correlation coefficient magnitude is: C_{ij} = |G_{ij}| / sqrt(D_i D_j).
    If C_{ij} > tau, the penalized error is the excess portion: 
        Error_{ij} = - G_{ij} * (1 - tau / C_{ij})
    If C_{ij} <= tau, Error_{ij} = 0.
    
    Gradient ∇_{X*} f ≈ 4 X @ Error (using standard Wirtinger derivative structure).

    x_ref : (B, 2, N_R, N_T) real/imag tensor
    tau   : threshold for correlation coefficient magnitude (default: 0.3)
    Returns gradient with the same shape as x_ref.
    """
    if x_ref.dim() != 4 or x_ref.size(1) != 2:
        raise ValueError("x_ref must have shape (batch, 2, N_R, N_T)")

    x_comp = torch.complex(x_ref[:, 0], x_ref[:, 1])              # (B, N_R, N_T)
    G = x_comp.conj().transpose(-1, -2) @ x_comp                   # (B, N_T, N_T)

    # Extract diagonal matrix D
    D_val = torch.diagonal(G, dim1=-2, dim2=-1).real               # (B, N_T)
    D_mat = torch.diag_embed(torch.complex(D_val, torch.zeros_like(D_val)))

    # Compute theoretical correlation denominator: sqrt(D_i * D_j)
    D_sqrt = torch.sqrt(torch.clamp(D_val, min=1e-8))              # (B, N_T)
    norm_matrix = D_sqrt.unsqueeze(-1) * D_sqrt.unsqueeze(-2)      # (B, N_T, N_T)

    # Extract off-diagonal part of G
    off_diag_G = G - D_mat

    # Calculate correlation magnitude
    corr_mag = off_diag_G.abs() / norm_matrix                      # (B, N_T, N_T)

    # Soft-threshold factor: 1 - tau / corr_mag. 
    # Clipped to [0, 1]. If corr_mag <= tau, factor is 0.
    excess_factor = torch.clamp(1.0 - tau / (corr_mag + 1e-8), min=0.0)

    # The error pushes the element back towards the tau boundary, not absolute zero
    error = - (off_diag_G * excess_factor)                         # (B, N_T, N_T)

    grad_comp = 4.0 * (x_comp @ error)                             # (B, N_R, N_T)

    return torch.stack((grad_comp.real, grad_comp.imag), dim=1)    # (B, 2, N_R, N_T)


def normalize_cov_grad(
    grad_cov: Tensor,
    x_ref: Tensor,
    cov: Tensor,
    norm_mode: Literal['none', 'by_x', 'by_r', 'global'] = 'none',
    eps: float = 1e-8,
) -> Tensor:
    """
    Normalize covariance gradient to stabilize its scale.
    
    Parameters
    ----------
    grad_cov : Tensor
        Covariance gradient, shape (batch, 2, n_meas, n_input)
    x_ref : Tensor
        Reference x, shape (batch, 2, n_meas, n_input)
    cov : Tensor
        Covariance matrix, shape (batch, 2, n_meas, n_meas)
    norm_mode : str
        Normalization mode:
        - 'none': No normalization (default, backward compatible)
        - 'by_x': Normalize by ||x||
        - 'by_r': Normalize by ||R|| (covariance matrix norm)
        - 'global': Normalize by batch mean of ||grad_cov||
    eps : float
        Small epsilon to avoid division by zero
    
    Returns
    -------
    grad_cov_normalized : Tensor
        Normalized gradient with same shape as grad_cov
    """
    if norm_mode == 'none':
        return grad_cov
    
    # Compute L2 norm over non-batch dimensions for each sample
    norm_dims = tuple(range(1, grad_cov.ndim))
    
    if norm_mode == 'by_x':
        # Normalize by ||x||
        x_norm = torch.linalg.vector_norm(x_ref, dim=norm_dims, keepdim=True)
        scale = x_norm + eps
    elif norm_mode == 'by_r':
        # Normalize by ||R|| (covariance matrix Frobenius norm)
        cov_comp = torch.complex(cov[:, 0], cov[:, 1])
        # Compute Frobenius norm for each sample: ||R||_F
        r_norm = torch.linalg.matrix_norm(cov_comp, ord='fro', dim=(-2, -1))  # (B,)
        # Expand to match grad_cov shape: (B, 1, 1, 1) -> (B, 2, n_meas, n_input)
        r_norm = r_norm.view(-1, 1, 1, 1).expand_as(grad_cov)
        scale = r_norm + eps
    elif norm_mode == 'global':
        # Normalize by batch mean of ||grad_cov|| to make ||grad|| ≈ 1
        grad_norm = torch.linalg.vector_norm(grad_cov, dim=norm_dims, keepdim=True)
        mean_grad_norm = grad_norm.mean()
        scale = mean_grad_norm + eps
        # Each sample is normalized by the batch mean, so ||grad|| ≈ 1 per sample
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")
    
    return grad_cov / scale
 


# ------------------------------- DPS sampler ----------------------------- #

LikelihoodGradFn = Callable[[Tensor, Tensor, int], Tensor]


class DpsSampler(object):
    """
    Diffusion Posterior Sampling (DPS) wrapper around an unconditional DiffusionModel.

    We follow the discrete-time spirit of DPS / classifier guidance:
        1) At each reverse step t, compute a likelihood gradient at the
           current noisy state x_t.
        2) Apply a small correction along this gradient.
        3) Feed the corrected x_t into the original reverse_step of the DM.

    This keeps the DM prior unchanged and adds a data-consistency term.
    """

    def __init__(
        self,
        dm,
        likelihood_grad_fn: LikelihoodGradFn,
        dps_lambda: float = 1.0,
        cov_lambda: float = 0.0,
        tx_cov_lambda: float = 0.0,  # ADDED parameter
        cov_scale_mode: Literal['beta_t', 'sqrt_beta_t', 'constant', 'snr_aware'] = 'beta_t',
        cov_beta_power: Optional[float] = None,
        cov_grad_norm: Literal['none', 'by_x', 'by_r', 'global'] = 'none',
        cov_step_clip: Optional[float] = None,
        cov_clip_mode: Literal['auto', 'elementwise', 'norm'] = 'auto',
        add_random: bool = True,
        H=None,
        sigma_y2: Optional[float] = None,
        lambda_dps: Optional[float] = None,
        exp_key: Literal['A', 'B', 'C', 'D', 'E', 'Eprime', 'F', 'G', 'H'] = 'A',
        gamma: float = 1.0,
        like_weight: float = 1.0,
        lw_schedule: Literal['const', 'ramp', 'lastk'] = 'const',
        lw_tau: float = 0.95,
        lw_max: float = 8.0,
        lw_end: float = 1.0,
        lw_k: int = 0,
        g_tau1: int = 0,
        like_beta_power: float = 1.0,
        like_snr_gate: bool = False,
        like_snr0_db: float = -10.5,
        like_snr_delta_db: float = 2.0,
        dps_scale_mode: Literal['beta_t', 'sigma_eff'] = 'beta_t',
        sigma_eff_c: float = 1.0,
        like_clip_mode: Literal['elementwise', 'norm'] = 'norm',
    ) -> None:
        """
        Parameters
        ----------
        dm : DiffusionModel
            Trained DMCE DiffusionModel.
        likelihood_grad_fn : callable
            g(x_ref, y, t) -> ∇_x log p(y | x_ref).
        dps_lambda : float
            Global guidance strength for DPS (used if lambda_dps is None).
        cov_lambda : float
            Covariance guidance strength.
        cov_scale_mode : str
            Scaling mode for covariance guidance:
            - 'beta_t': zeta_t = beta_t (default, backward compatible)
            - 'sqrt_beta_t': zeta_t = sqrt(beta_t)
            - 'constant': zeta_t = 1.0
            - 'snr_aware': zeta_t = 1/sqrt(beta_t) (may need clipping)
        cov_beta_power : float, optional
            If set, overrides cov_scale_mode and uses zeta_t = beta_t**p for covariance guidance scaling.
            Examples: p=1.0 -> beta_t, p=0.5 -> sqrt_beta_t, p=0.0 -> constant, p=-0.5 -> snr_aware-like.
        cov_grad_norm : str
            Normalization mode for covariance gradient:
            - 'none': No normalization (default, backward compatible)
            - 'by_x': Normalize by ||x||
            - 'by_r': Normalize by ||R|| (covariance matrix norm)
            - 'global': Normalize by batch mean of ||grad_cov||
        cov_step_clip : float, optional
            Separate clipping threshold for covariance correction.
            If None, uses self.step_clip (backward compatible).
        cov_clip_mode : str
            Clipping mode for covariance correction:
            - 'auto' (default): backward-compatible behavior (elementwise for legacy beta_t path, norm otherwise)
            - 'elementwise': element-wise clamp to [-C, C]
            - 'norm': L2-norm-based scaling clip (preserves direction)
        like_clip_mode : str
            How to clip the DPS likelihood correction before adding to ``x_prior``:
            - 'norm' (default): per-batch-sample L2 cap on flattened correction; same rescaling idea as ``cov_clip_mode='norm'``.
            - 'elementwise': clamp each tensor element to ``[-step_clip, step_clip]``.
        add_random : bool
            Whether to use stochastic reverse steps (same meaning as in DMCE).
        H, sigma_y2 :
            Kept for API compatibility with DMCE, but not explicitly used
            in this AWGN-only implementation.
        lambda_dps : float, optional
            Some DMCE code passes this name instead of dps_lambda.
            If provided, it overrides dps_lambda.
        """
        self.dm = dm
        self.device = dm.device
        self.likelihood_grad_fn = likelihood_grad_fn

        if lambda_dps is not None:
            self.dps_lambda = float(lambda_dps)
        else:
            self.dps_lambda = float(dps_lambda)

        self.cov_lambda = float(cov_lambda)
        self.tx_cov_lambda = float(tx_cov_lambda)
        self.cov_lambda_base = float(cov_lambda)  # Store base value for t_start scaling
        self.cov_scale_mode = cov_scale_mode
        self.cov_beta_power = None if cov_beta_power is None else float(cov_beta_power)
        self.cov_grad_norm = cov_grad_norm
        self.cov_step_clip = cov_step_clip
        self.cov_clip_mode = str(cov_clip_mode)
        self.add_random = add_random
        self.H = H
        self.sigma_y2 = sigma_y2
        self.use_t_start_scaling = False  # Flag to enable t_start-based scaling
        self.debug_cov_scaling = False  # Flag to enable debug logging for cov scaling
        self.debug_history = []  # Store debug info for each timestep

        # Safety knobs for numerical stability
        # Clip the final likelihood correction (not the gradient); see like_clip_mode (norm vs elementwise).
        self.grad_clip = None        # Do NOT clip gradients (they can be large, that's OK)
        self.step_clip = 2.0         # Max L2 norm per sample (norm mode) or elementwise bound (elementwise mode)

        # Ablation experiment key: A=baseline, B=scalar Jacobian, C=autograd gold, D=soft-gated
        self.exp_key = exp_key
        self.gamma = float(gamma)  # For experiment D: soft-gated likelihood guidance
        self.debug_likelihood = False  # Enable debug prints for likelihood guidance
        self.debug_cf = False  # Enable closed-form likelihood debug / micro-test mode (exp E)
        self.like_weight = float(like_weight)  # For exp E: scale closed-form likelihood score
        # Time-dependent likelihood weight schedule (exp E)
        self.lw_schedule = str(lw_schedule)
        self.lw_tau = float(lw_tau)
        self.lw_max = float(lw_max)
        self.lw_end = float(lw_end)
        self.lw_k = int(lw_k)
        # Exp G (paper Algorithm-3 style) loop cutoff: stop at t = g_tau1 (>=1 recommended).
        # If g_tau1 == 0, run all the way to t=0 (may be unstable for Alg-3 correction).
        self.g_tau1 = int(g_tau1)
        # Diagnostic: relative likelihood-vs-prior strength (exp E)
        self.record_like_balance = False
        self.like_balance_history = []
        
        # Debug control: printing vs micro-test (3-step) mode
        self.debug_cf_microtest = False
        # Power applied to beta_t for post-add likelihood correction (A/B/C/D only).
        # Default=1.0 preserves existing behavior.
        self.like_beta_power = float(like_beta_power)

        # EXP H option: sigmoid gate on likelihood strength based on observation SNR (in dB).
        # gate(SNR_dB) = 1 / (1 + exp(-(SNR_dB - SNR0)/Delta))
        self.like_snr_gate = bool(like_snr_gate)
        self.like_snr0_db = float(like_snr0_db)
        self.like_snr_delta_db = float(like_snr_delta_db)
        self._like_gate_cached: Optional[float] = None
        # Post-add likelihood scalar: 'beta_t' (default) vs effective-variance schedule.
        self.dps_scale_mode = str(dps_scale_mode)
        if self.dps_scale_mode not in ('beta_t', 'sigma_eff'):
            raise ValueError("dps_scale_mode must be 'beta_t' or 'sigma_eff'.")
        self.sigma_eff_c = float(sigma_eff_c)
        self.like_clip_mode = str(like_clip_mode)
        if self.like_clip_mode not in ('elementwise', 'norm'):
            raise ValueError("like_clip_mode must be 'elementwise' or 'norm'.")

        # Extract sigma_y2 from likelihood_grad_fn closure for experiment C
        # This is needed for the autograd loss computation
        if exp_key == 'C':
            if hasattr(likelihood_grad_fn, '__closure__') and likelihood_grad_fn.__closure__:
                self.sigma_y2_for_exp_c = likelihood_grad_fn.__closure__[0].cell_contents
            elif sigma_y2 is not None:
                self.sigma_y2_for_exp_c = sigma_y2
            else:
                # Fallback: try to extract from function name or use default
                self.sigma_y2_for_exp_c = 1.0
                import warnings
                warnings.warn("Could not extract sigma_y2 for experiment C, using default 1.0")
        else:
            self.sigma_y2_for_exp_c = None

    def reverse_step_dps(self, x_t: Tensor, y: Tensor, cov: Tensor, t: int, diagnostic_recorder=None) -> Tensor:
        """
        Single DPS reverse step at discrete time index t.

        DPS theory: Apply gradient correction AFTER the prior reverse step.
        
        1) First, do the unconditional prior reverse step: x_prior = reverse_step(x_t, t)
        2) Compute likelihood gradient at x_prior (not at x_t)
        3) Apply correction: x_prev = x_prior + eta_t * grad_like(x_prior, y)
        
        This ensures the gradient is evaluated at the denoised estimate, not the noisy state.
        
        Ablation experiments:
        - A: Baseline (ignore Jacobian)
        - B: Scalar Jacobian approximation (1/sqrt(alpha_bar_t))
        - C: Autograd gold gradient (full Jacobian via autograd)
        - D: Soft-gated scaling on the likelihood update
        - E: Closed-form likelihood score, injected into reverse-mean: score_total = score_prior + w*score_like_cf
        - Eprime: diagnostic post-add using sigma_t^2 * score_like_cf (no beta_t)
        - F: Paper-style reverse optimization update: H_t <- H_t + post_var_t * score_post, then noise injection
        - G: Paper Algorithm-3 style: DDIM deterministic step + explicit posterior correction term
        - H: "Classic DPS" likelihood: form x0_hat(H_t) from prior score, then use grad_like ≈ (y - x0_hat)/sigma^2 as ∇_{H_t} log p(y|H_t)
        """
        # For experiments A and B, we can use no_grad context
        # For experiment C, we need gradients enabled
        if self.exp_key == 'C':
            if getattr(self, "_gaussian_pilot_ctx", None) is not None:
                raise NotImplementedError(
                    "Gaussian-pilot likelihood is not implemented for exp_key='C' (autograd gold gradient)."
                )
            # Experiment C: Autograd "gold" gradient
            # Enable gradients for x_t
            x_t_requires_grad = x_t.clone().requires_grad_(True)
            
            # Compute H0_hat(Ht) with gradients enabled
            # NOTE: self.dm.reverse_step() is decorated with @torch.no_grad(),
            # so we need to manually implement the reverse step logic here
            with torch.enable_grad():
                # Manually implement reverse_step without @torch.no_grad() decorator
                # This is a copy of the logic from dm.reverse_step() but without no_grad
                b, *_ = x_t_requires_grad.shape
                batched_times = torch.full((b,), t, device=self.dm.device, dtype=torch.long)
                
                if self.dm.reverse_method == 'reverse_mean':
                    if self.dm.objective == 'pred_noise':
                        pred_noise = self.dm.model(x_t_requires_grad, batched_times)
                        # Manually compute posterior mean from noise (differentiable version)
                        # Formula from get_posterior_mean_from_noise: 
                        # posterior_mean = sqrt_recip_alphas * x_t - post_mean_from_noise_coef * noise
                        sqrt_recip_alphas = dm_utils.extract(self.dm.sqrt_recip_alphas, batched_times, x_t_requires_grad.shape)
                        post_mean_from_noise_coef = dm_utils.extract(self.dm.post_mean_from_noise_coef, batched_times, x_t_requires_grad.shape)
                        x_prior_grad = sqrt_recip_alphas * x_t_requires_grad - post_mean_from_noise_coef * pred_noise
                    elif self.dm.objective == 'pred_x_0':
                        x_0_pred = self.dm.model(x_t_requires_grad, batched_times)
                        # Manually compute posterior mean from x_0 (differentiable version)
                        # Formula from get_posterior_mean_from_x_0:
                        # posterior_mean = posterior_mean_coef_x_0 * x_0 + posterior_mean_coef_x_t * x_t
                        posterior_mean_coef_x_0 = dm_utils.extract(self.dm.posterior_mean_coef_x_0, batched_times, x_0_pred.shape)
                        posterior_mean_coef_x_t = dm_utils.extract(self.dm.posterior_mean_coef_x_t, batched_times, x_t_requires_grad.shape)
                        x_prior_grad = posterior_mean_coef_x_0 * x_0_pred + posterior_mean_coef_x_t * x_t_requires_grad
                    elif self.dm.objective == 'pred_post_mean':
                        x_prior_grad = self.dm.model(x_t_requires_grad, batched_times)
                    else:
                        raise ValueError(f'Objective {self.dm.objective} is not supported.')
                    
                    # For deterministic steps (add_random=False), x_pred = posterior_mean
                    # No need to add noise
                else:
                    raise ValueError(f'Reverse method {self.dm.reverse_method} is not supported for experiment C.')
                
                # Compute loss: (1/(2*sigma_y2)) * ||Y - H0_hat(Ht)||_F^2
                sigma_y2 = self.sigma_y2_for_exp_c
                diff = y - x_prior_grad
                loss_like = (1.0 / (2.0 * sigma_y2)) * torch.sum(diff ** 2)
                
                # Compute gradient: gradHt = -d(loss_like)/d(Ht)
                grad_like = -torch.autograd.grad(
                    loss_like, 
                    x_t_requires_grad, 
                    retain_graph=False, 
                    create_graph=False,
                    only_inputs=True
                )[0]
            
            # Detach to avoid graph growth
            grad_like = grad_like.detach()
            x_prior = x_prior_grad.detach()
        else:
            # Experiments A/B/D/E/Eprime/F: use no_grad context
            with torch.no_grad():
                if self.exp_key == 'F':
                    # ------------------------------------------------------------
                    # Exp F: paper-style reverse optimization update (Algorithm-2 style):
                    #   H_t'    = H_t + post_var_t * score_post
                    #   H_{t-1} = H_t' + sqrt(post_var_t) * z
                    # where score_post = score_prior + like_weight * score_like_cf.
                    # ------------------------------------------------------------
                    b, *_ = x_t.shape
                    batched_times = torch.full((b,), t, device=self.dm.device, dtype=torch.long)

                    if self.dm.objective != 'pred_noise':
                        raise NotImplementedError("Exp F currently supports only objective='pred_noise'.")

                    eps_hat = self.dm.model(x_t, batched_times)
                    a = torch.clamp(self.dm.alphas_cumprod[t].to(x_t.device), min=1e-12)
                    shape_ones = (1,) * (x_t.ndim - 1)
                    a_view = a.view(1, *shape_ones)
                    one_minus_a = torch.clamp(1.0 - a_view, min=1e-12)
                    score_prior = -eps_hat / torch.sqrt(one_minus_a)

                    sqrt_a = torch.sqrt(a_view)
                    sigma2 = torch.tensor(float(self.sigma_y2), device=x_t.device, dtype=x_t.dtype)
                    denom = sqrt_a * (sigma2 + (1.0 - a_view) / a_view)
                    if not torch.isfinite(denom).all():
                        raise RuntimeError(f"[exp F] Non-finite denom at t={t}: denom={denom}")
                    score_like_cf = (y - (x_t / sqrt_a)) / denom

                    w = torch.tensor(self.like_weight, device=x_t.device, dtype=x_t.dtype).view(1, *shape_ones)
                    score_post = score_prior + w * score_like_cf

                    post_var_t = dm_utils.extract(self.dm.posterior_variance, batched_times, x_t.shape)
                    x_t_map = x_t + post_var_t * score_post

                    noise = self.dm.noise_multiplier * torch.randn_like(x_t) if t > 0 else 0.0
                    x_prev = x_t_map + torch.sqrt(post_var_t) * noise

                    # For compatibility with the rest of the function
                    x_prior = x_t_map
                    grad_like_H0 = self.likelihood_grad_fn(x_prior, y, t)
                    grad_like = torch.zeros_like(x_prior)
                    correction = torch.zeros_like(x_prior)

                    if self.debug_cf and t in (self.dm.num_timesteps - 1, self.dm.num_timesteps // 2 - 1, 0):
                        def _bnorm(z: Tensor) -> float:
                            return torch.linalg.vector_norm(z, dim=tuple(range(1, z.ndim))).mean().item()

                        beta_val = float(self.dm.betas[t].item())
                        post_var_val = float(post_var_t.flatten()[0].item())
                        a_val = float(a.item())
                        eff_like_step = post_var_t * (w * score_like_cf)
                        state_change = x_prev - x_t
                        print(f"[DEBUG_CF F t={t}] beta_t={beta_val:.6e}, alpha_bar_t={a_val:.6e}, post_var_t={post_var_val:.6e}, like_weight={self.like_weight:.3f}")
                        print(f"  ||score_prior||_2={_bnorm(score_prior):.6e}")
                        print(f"  ||score_like_cf||_2={_bnorm(score_like_cf):.6e}")
                        print(f"  ||score_post||_2={_bnorm(score_post):.6e}")
                        print(f"  ||eff_like_step||_2={_bnorm(eff_like_step):.6e}   (=||post_var_t * like_weight * score_like_cf||)")
                        print(f"  ||H_{t-1}-H_t||_2={_bnorm(state_change):.6e}")
                        print()

                elif self.exp_key == 'E':
                    # Required: use full reverse, no SNR matching (handled by caller).
                    # 1) Compute score_prior in H_t-space from denoiser output (eps_hat).
                    b, *_ = x_t.shape
                    batched_times = torch.full((b,), t, device=self.dm.device, dtype=torch.long)

                    if self.dm.reverse_method != 'reverse_mean' or self.dm.objective != 'pred_noise':
                        raise NotImplementedError(
                            "Exp E currently supports only reverse_method='reverse_mean' with objective='pred_noise'."
                        )

                    eps_hat = self.dm.model(x_t, batched_times)
                    a = torch.clamp(self.dm.alphas_cumprod[t].to(x_t.device), min=1e-12)
                    shape_ones = (1,) * (x_t.ndim - 1)
                    a_view = a.view(1, *shape_ones)
                    sqrt_a = torch.sqrt(a_view)
                    one_minus_a = torch.clamp(1.0 - a_view, min=1e-12)

                    # DDPM score (approx): score_prior = -eps_hat / sqrt(1 - alpha_bar_t)
                    score_prior = -eps_hat / torch.sqrt(one_minus_a)

                    # 2) Closed-form likelihood score (tildeH = y, H_t = x_t)
                    sigma2 = torch.tensor(float(self.sigma_y2), device=x_t.device, dtype=x_t.dtype)
                    denom = sqrt_a * (sigma2 + (1.0 - a_view) / a_view)
                    if not torch.isfinite(denom).all():
                        raise RuntimeError(f"[exp E] Non-finite denom at t={t}: denom={denom}")
                    score_like_cf = (y - (x_t / sqrt_a)) / denom

                    # 3) score_total = score_prior + w_t * score_like_cf
                    # w_t can be constant (like_weight) or a gated linear ramp based on a = alpha_bar_t.
                    if self.lw_schedule == 'const':
                        w_scalar = float(self.like_weight)
                        w_t = torch.tensor(w_scalar, device=x_t.device, dtype=x_t.dtype)
                    elif self.lw_schedule == 'ramp':
                        # w_t = 1 + (w_max - 1) * clip((a - tau)/(1 - tau), 0, 1)
                        a_clip = torch.clamp(a, 0.0, 1.0)
                        tau = float(self.lw_tau)
                        w_max = float(self.lw_max)
                        den = max(1.0 - tau, 1e-12)
                        ramp = torch.clamp((a_clip - tau) / den, 0.0, 1.0)
                        w_t = 1.0 + (w_max - 1.0) * ramp
                        w_t = w_t.to(device=x_t.device, dtype=x_t.dtype)
                    elif self.lw_schedule == 'lastk':
                        # w_t = lw_end for t <= lw_k, else 1.0
                        w_end = float(self.lw_end)
                        k = int(self.lw_k)
                        w_t = torch.tensor((w_end if int(t) <= k else 1.0), device=x_t.device, dtype=x_t.dtype)
                    else:
                        raise ValueError(f"Unknown lw_schedule: {self.lw_schedule}. Use 'const', 'ramp', or 'lastk'.")
                    w = w_t.view(1, *shape_ones)
                    score_total = score_prior + w * score_like_cf

                    # 4) Use score_total inside the SAME reverse-mean formula.
                    # Reverse-mean can be written as: mu = 1/sqrt(alpha_t) * (x_t + beta_t * score_total)
                    beta_t = self.dm.betas[t].to(x_t.device).view(1, *shape_ones)
                    alpha_t = (1.0 - self.dm.betas[t]).to(x_t.device).view(1, *shape_ones)
                    mu = (x_t + beta_t * score_total) / torch.sqrt(alpha_t + 1e-12)

                    # ------------------------------------------------------------
                    # Diagnostic logging (no behavior change):
                    #   P_t = ||beta_t * score_prior||_2
                    #   L_t = ||beta_t * w_t * score_like_cf||_2
                    #   r_t = L_t / (P_t + eps)
                    # ------------------------------------------------------------
                    if self.record_like_balance:
                        def _bnorm(z: Tensor) -> float:
                            return torch.linalg.vector_norm(z, dim=tuple(range(1, z.ndim))).mean().item()

                        eps = 1e-12
                        P_t = _bnorm(beta_t * score_prior)
                        L_t = _bnorm(beta_t * w * score_like_cf)
                        r_t = L_t / (P_t + eps)
                        # Optional additional ratio (independent of beta_t)
                        P0 = _bnorm(score_prior)
                        L0 = _bnorm(w * score_like_cf)
                        r0 = L0 / (P0 + eps)

                        self.like_balance_history.append({
                            't': int(t),
                            'alpha_bar_t': float(a.item()),
                            'beta_t': float(beta_t.flatten()[0].item()),
                            'like_weight': float(self.like_weight),
                            'lw_schedule': str(self.lw_schedule),
                            'lw_tau': float(self.lw_tau),
                            'lw_max': float(self.lw_max),
                            'w_t': float(w_t.item()) if isinstance(w_t, torch.Tensor) else float(w_t),
                            'P_t': float(P_t),
                            'L_t': float(L_t),
                            'r_t': float(r_t),
                            'P0': float(P0),
                            'L0': float(L0),
                            'r0': float(r0),
                        })

                    if self.add_random:
                        # Same noise injection as baseline reverse_step
                        posterior_variance = dm_utils.extract(self.dm.posterior_variance, batched_times, x_t.shape)
                        noise = self.dm.noise_multiplier * torch.randn_like(x_t) if t > 0 else 0.0
                        x_prior = mu + torch.sqrt(posterior_variance) * noise
                    else:
                        x_prior = mu

                    # For exp E, likelihood is already injected into x_prior; set correction=0
                    grad_like_H0 = self.likelihood_grad_fn(x_prior, y, t)
                    grad_like = torch.zeros_like(x_prior)
                    correction = torch.zeros_like(x_prior)

                    # Debug prints at t=99,49,0 (or whatever T is)
                    if self.debug_cf and t in (self.dm.num_timesteps - 1, self.dm.num_timesteps // 2 - 1, 0):
                        def _bnorm(z: Tensor) -> float:
                            return torch.linalg.vector_norm(z, dim=tuple(range(1, z.ndim))).mean().item()

                        # Show effective contribution to mean update term: beta_t * w * score_like_cf
                        eff_like = beta_t * w * score_like_cf / torch.sqrt(alpha_t + 1e-12)
                        print(f"[DEBUG_CF t={t}] beta_t={float(beta_t.flatten()[0].item()):.6e}, alpha_bar_t={float(a.item()):.6e}, like_weight={self.like_weight:.3f}")
                        print(f"  ||score_prior||_2={_bnorm(score_prior):.6e}")
                        print(f"  ||score_like_cf||_2={_bnorm(score_like_cf):.6e}")
                        print(f"  ||score_total||_2={_bnorm(score_total):.6e}")
                        print(f"  ||eff_like_in_mu||_2={_bnorm(eff_like):.6e}   (should NOT be ~1e-4 at t=0 unless score_like is tiny)")
                elif self.exp_key == 'G':
                    # ------------------------------------------------------------
                    # Exp G: Paper Algorithm-3 style (DDIM-based posterior correction)
                    #
                    # Step 3 (DDIM deterministic, eta=0):
                    #   x0_hat   = (x_t - sqrt(1-a_t)*eps_hat) / sqrt(a_t)
                    #   x_{t-1}  = sqrt(a_{t-1})*x0_hat + sqrt(1-a_{t-1})*eps_hat
                    #
                    # Step 4 (closed-form likelihood gradient wrt x_t):
                    #   score_like_cf = (y - x_t/sqrt(a_t)) / ( sqrt(a_t) * (sigma2 + (1-a_t)/a_t) )
                    #
                    # Step 5 (paper operator in front of likelihood):
                    #   c_t = sqrt(1-a_{t-1}) - sqrt(a_{t-1}*(1-a_t))/sqrt(a_t)
                    #   x_{t-1} <- x_{t-1} + dps_lambda * c_t * score_like_cf
                    # ------------------------------------------------------------
                    if self.add_random:
                        raise RuntimeError("Exp G requires deterministic steps (add_random=False), matching DDIM-based Algorithm 3.")

                    b, *_ = x_t.shape
                    batched_times = torch.full((b,), t, device=self.dm.device, dtype=torch.long)

                    if self.dm.objective != 'pred_noise':
                        raise NotImplementedError("Exp G currently supports only objective='pred_noise'.")

                    # eps_theta(x_t, t)
                    eps_hat = self.dm.model(x_t, batched_times)

                    # alpha_bar_t and alpha_bar_{t-1}
                    a_t = torch.clamp(self.dm.alphas_cumprod[t].to(x_t.device), min=1e-12)
                    a_prev = torch.clamp(self.dm.alphas_cumprod[t - 1].to(x_t.device), min=1e-12) if t > 0 else None

                    shape_ones = (1,) * (x_t.ndim - 1)
                    a_t_view = a_t.view(1, *shape_ones)
                    sqrt_a_t = torch.sqrt(a_t_view)
                    sqrt_one_minus_a_t = torch.sqrt(torch.clamp(1.0 - a_t_view, min=1e-12))

                    # x0_hat and DDIM x_{t-1}
                    x0_hat = (x_t - sqrt_one_minus_a_t * eps_hat) / torch.sqrt(a_t_view + 1e-12)

                    if t == 0:
                        # Already at the last step: output x0_hat (no t-1 exists)
                        x_prior = x0_hat
                        grad_like_H0 = self.likelihood_grad_fn(x_prior, y, t)
                        grad_like = torch.zeros_like(x_prior)
                        correction = torch.zeros_like(x_prior)
                    else:
                        a_prev_view = a_prev.view(1, *shape_ones)
                        x_ddim = torch.sqrt(a_prev_view) * x0_hat + torch.sqrt(torch.clamp(1.0 - a_prev_view, min=1e-12)) * eps_hat

                        # closed-form likelihood gradient wrt x_t (tildeH=y)
                        sigma2 = torch.tensor(float(self.sigma_y2), device=x_t.device, dtype=x_t.dtype)
                        denom = sqrt_a_t * (sigma2 + (1.0 - a_t_view) / a_t_view)
                        if not torch.isfinite(denom).all():
                            raise RuntimeError(f"[exp G] Non-finite denom at t={t}: denom={denom}")
                        score_like_cf = (y - (x_t / sqrt_a_t)) / denom

                        # Paper Algorithm-3 Step-5 coefficient:
                        # The screenshot shows an additional normalization by
                        #   denom_step5 = sqrt(a_{t-1}(1-a_t)) / sqrt(a_t)
                        # so the effective coefficient is:
                        #   coef = (sqrt(1-a_{t-1}) - denom_step5) / denom_step5
                        #       = sqrt(1-a_{t-1})/denom_step5 - 1
                        denom_step5 = torch.sqrt(torch.clamp(a_prev_view * (1.0 - a_t_view), min=0.0)) / torch.sqrt(a_t_view + 1e-12)
                        numer_step5 = torch.sqrt(torch.clamp(1.0 - a_prev_view, min=0.0)) - denom_step5
                        coef_step5 = numer_step5 / torch.clamp(denom_step5, min=1e-12)

                        x_prior = x_ddim + self.dps_lambda * coef_step5 * score_like_cf

                        # Keep API-compatible placeholders (no post-add correction path for G)
                        grad_like_H0 = self.likelihood_grad_fn(x_prior, y, t)
                        grad_like = torch.zeros_like(x_prior)
                        correction = torch.zeros_like(x_prior)

                        # For Exp G, instability (if any) almost always happens at very late steps.
                        # So in debug mode, print at a few canonical steps plus all t<=5.
                        if self.debug_cf and (t in (self.dm.num_timesteps - 1, self.dm.num_timesteps // 2 - 1) or int(t) <= 5):
                            def _bnorm(z: Tensor) -> float:
                                return torch.linalg.vector_norm(z, dim=tuple(range(1, z.ndim))).mean().item()

                            upd = self.dps_lambda * coef_step5 * score_like_cf
                            print(f"[DEBUG_G t={t}] alpha_bar_t={float(a_t.item()):.6e}, alpha_bar_prev={float(a_prev.item()):.6e}, dps_lambda={float(self.dps_lambda):.3f}")
                            print(f"  denom_step5={float(denom_step5.flatten()[0].item()):.6e}")
                            print(f"  coef_step5={float(coef_step5.flatten()[0].item()):.6e}")
                            print(f"  sigma2={float(sigma2.item()):.6e}")
                            print(f"  ||x_ddim||_2={_bnorm(x_ddim):.6e}")
                            print(f"  ||score_like_cf||_2={_bnorm(score_like_cf):.6e}")
                            print(f"  ||update||_2={_bnorm(upd):.6e}   (=||dps_lambda * coef_step5 * score_like_cf||)")
                else:
                    # 1) Unconditional prior reverse step
                    x_prior = self.dm.reverse_step(x_t, t, add_random=self.add_random)

                    # 2) Likelihood direction for guidance
                    gp_ctx = getattr(self, "_gaussian_pilot_ctx", None)
                    if gp_ctx is not None:
                        if self.exp_key in ("E", "Eprime", "F", "G", "C"):
                            raise NotImplementedError(
                                "Spatial-pilot DPS supports exp_key in {'A','B','D','H'} only; "
                                "closed-form variants E/E'/F/G and autograd C are unsupported."
                            )
                        if gp_ctx.get("pilot_likelihood_domain") == "angular_ls":
                            grad_like_H0 = angular_ls_mahalanobis_likelihood_grad_angular(
                                x_t,
                                gp_ctx,
                                self.dm,
                                t,
                                float(self.sigma_y2),
                            )
                        elif gp_ctx.get("pilot_likelihood_mode") == "nonorthogonal":
                            grad_like_H0 = nonorthogonal_pilot_likelihood_grad_angular(
                                x_t,
                                gp_ctx,
                                self.dm,
                                t,
                                float(self.sigma_y2),
                            )
                        else:
                            grad_like_H0 = gaussian_pilot_likelihood_grad_angular(
                                x_t,
                                gp_ctx,
                                self.dm,
                                t,
                                float(self.sigma_y2),
                            )
                    elif self.exp_key == 'H':
                        # ------------------------------------------------------------
                        # Exp H: "classic DPS" likelihood gradient in H_t-space.
                        #
                        # Approximate:
                        #   ∇_{H_t} log p(y | H_t) ≈ (y - x0_hat(H_t)) / sigma_y^2
                        # where x0_hat(H_t) is formed from the prior score:
                        #   score_prior(H_t) = ∇_{H_t} log p(H_t)
                        #   x0_hat = 1/sqrt(alpha_bar_t) * (H_t + (1 - alpha_bar_t) * score_prior(H_t))
                        #
                        # For objective='pred_noise', we use:
                        #   score_prior ≈ -eps_hat / sqrt(1 - alpha_bar_t)
                        # which makes x0_hat identical to the standard DDPM formula:
                        #   x0_hat = (H_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
                        # ------------------------------------------------------------
                        if self.dm.objective != 'pred_noise':
                            raise NotImplementedError("Exp H currently supports only objective='pred_noise'.")
                        b, *_ = x_t.shape
                        batched_times = torch.full((b,), t, device=self.dm.device, dtype=torch.long)
                        eps_hat = self.dm.model(x_t, batched_times)
                        a = torch.clamp(self.dm.alphas_cumprod[t].to(x_t.device), min=1e-12)
                        shape_ones = (1,) * (x_t.ndim - 1)
                        a_view = a.view(1, *shape_ones)
                        one_minus_a = torch.clamp(1.0 - a_view, min=1e-12)
                        # score_prior(H_t)
                        score_prior = -eps_hat / torch.sqrt(one_minus_a)
                        # x0_hat = (H_t + (1-a)*score_prior) / sqrt(a)
                        x0_hat = (x_t + one_minus_a * score_prior) / torch.sqrt(a_view + 1e-12)
                        grad_like_H0 = self.likelihood_grad_fn(x0_hat, y, t)
                    else:
                        grad_like_H0 = self.likelihood_grad_fn(x_prior, y, t)

                    if self.exp_key == 'A':
                        grad_like = grad_like_H0
                    elif self.exp_key == 'B':
                        alpha_bar_t = self.dm.alphas_cumprod[t]
                        jacobian_scalar = 1.0 / torch.sqrt(alpha_bar_t + 1e-12)
                        shape_ones = (1,) * (x_prior.ndim - 1)
                        jacobian_scalar = jacobian_scalar.view(1, *shape_ones).to(x_prior.device)
                        grad_like = jacobian_scalar * grad_like_H0
                    elif self.exp_key == 'D':
                        grad_like = grad_like_H0
                    elif self.exp_key == 'Eprime':
                        # Eprime diagnostic: post-add structure using sigma_t^2 instead of beta_t.
                        # score_like_cf uses tildeH=y and H_t=x_t (paper-style closed-form).
                        a = torch.clamp(self.dm.alphas_cumprod[t].to(x_prior.device), min=1e-12)
                        shape_ones = (1,) * (x_prior.ndim - 1)
                        a_view = a.view(1, *shape_ones)
                        sqrt_a = torch.sqrt(a_view)
                        sigma2 = torch.tensor(float(self.sigma_y2), device=x_prior.device, dtype=x_prior.dtype)
                        denom = sqrt_a * (sigma2 + (1.0 - a_view) / a_view)
                        score_like_cf = (y - (x_t / sqrt_a)) / denom
                        grad_like = score_like_cf  # treat as score-like term for diagnostic
                    elif self.exp_key == 'H':
                        # Use the "classic" approximate likelihood gradient in H_t-space
                        grad_like = grad_like_H0
                    else:
                        raise ValueError(f"Unknown exp_key: {self.exp_key}. Must be 'A', 'B', 'C', 'D', 'E', 'Eprime', 'F', 'G', or 'H'.")

        # 3) Compute likelihood correction (post-add) for A/B/D/Eprime/C.
        # Exp E uses score injection into x_prior above, so correction is already set to 0.
        # Exp F uses paper-style update, so correction is also 0.
        if self.exp_key not in ('E', 'F', 'G'):
            with torch.no_grad():
                beta_t = self.dm.betas[t]  # scalar
                shape_ones = (1,) * (x_prior.ndim - 1)  # broadcast over non-batch dims
                alpha_bar_t = torch.clamp(
                    self.dm.alphas_cumprod[t].to(x_prior.device), min=1e-12
                )

                # Base scalar for likelihood correction: beta_t (default) or sigma_eff schedule.
                # sigma_eff^2(t) = sigma_y2 + c * (1 - alpha_bar_t) inflates the denominator vs sigma_y2 alone
                # so guidance does not explode when (1-alpha_bar_t) is large. grad_like already includes 1/sigma_y2
                # (AWGN score), so we multiply by sigma_y2/sigma_eff^2 — net coefficient on (y - x_ref) is 1/sigma_eff^2,
                # not 1/(sigma_y2 * sigma_eff^2).
                if self.dps_scale_mode == 'sigma_eff':
                    s2 = float(self.sigma_y2) if self.sigma_y2 is not None else 1.0
                    s2_t = torch.tensor(s2, device=x_prior.device, dtype=x_prior.dtype)
                    c_t = torch.tensor(float(self.sigma_eff_c), device=x_prior.device, dtype=x_prior.dtype)
                    sigma_eff2 = s2_t + c_t * (1.0 - alpha_bar_t)
                    base_scale = (s2_t / torch.clamp(sigma_eff2, min=1e-10)).view(1, *shape_ones)
                else:
                    beta_like = beta_t
                    if self.like_beta_power != 1.0 and self.exp_key in ('A', 'B', 'C', 'D', 'H'):
                        beta_like = torch.pow(beta_t, self.like_beta_power)
                    base_scale = beta_like.view(1, *shape_ones).to(x_prior.device)

                if self.exp_key == 'A':
                    like_scalar_t = base_scale
                elif self.exp_key == 'B':
                    like_scalar_t = base_scale
                elif self.exp_key == 'H':
                    like_scalar_t = base_scale
                elif self.exp_key == 'D':
                    exponent = self.gamma - 0.5
                    like_scalar_t = base_scale * torch.pow(alpha_bar_t, exponent)
                elif self.exp_key == 'Eprime':
                    # Diagnostic: sigma_t^2 scaling instead of beta_t (unchanged by dps_scale_mode).
                    a = torch.clamp(self.dm.alphas_cumprod[t].to(x_prior.device), min=1e-12)
                    sigma_t2 = ((1.0 - a) / a).view(1, *shape_ones).to(x_prior.device)
                    like_scalar_t = sigma_t2
                else:
                    # Experiment C: autograd gradient; same outer scalar as A (beta_t or sigma_eff schedule).
                    like_scalar_t = base_scale

                correction = self.dps_lambda * like_scalar_t * grad_like

                # EXP H optional SNR-based sigmoid gate (constant per run):
                # w_like(t, SNR_dB) = lambda_like * beta_t * gate(SNR_dB)
                if (
                    self.exp_key == 'H'
                    and self.like_snr_gate
                    and getattr(self, "_gaussian_pilot_ctx", None) is None
                ):
                    gate = 1.0 if self._like_gate_cached is None else float(self._like_gate_cached)
                    gate_t = torch.tensor(gate, device=x_prior.device, dtype=x_prior.dtype).view(1, *shape_ones)
                    correction = correction * gate_t
            
            # Debug mode: print detailed information for micro-test
            if self.debug_likelihood:
                alpha_bar_t_val = self.dm.alphas_cumprod[t].item() if self.exp_key in ['B', 'D'] else None
                like_scalar_val = like_scalar_t.mean().item() if isinstance(like_scalar_t, torch.Tensor) else like_scalar_t
                grad_like_norm = torch.linalg.vector_norm(grad_like, dim=tuple(range(1, grad_like.ndim))).mean().item()
                correction_norm = torch.linalg.vector_norm(correction, dim=tuple(range(1, correction.ndim))).mean().item()
                
                print(f"[DEBUG t={t}] exp_key={self.exp_key}, gamma={self.gamma if self.exp_key == 'D' else 'N/A'}, dps_scale_mode={self.dps_scale_mode}")
                if self.dps_scale_mode == 'sigma_eff' and self.exp_key != 'Eprime':
                    s2d = float(self.sigma_y2) if self.sigma_y2 is not None else 1.0
                    ab = float(self.dm.alphas_cumprod[t].item())
                    se2 = s2d + float(self.sigma_eff_c) * (1.0 - ab)
                    ratio = s2d / se2
                    print(
                        f"  sigma_eff^2={se2:.6e} (= sigma_y2 + c*(1-alpha_bar)), "
                        f"like_scalar=sigma_y2/sigma_eff^2={ratio:.6e} (net residual scale 1/sigma_eff^2)"
                    )
                if alpha_bar_t_val is None:
                    print(f"  beta_t={beta_t.item():.6e}, alpha_bar_t=N/A")
                else:
                    print(f"  beta_t={beta_t.item():.6e}, alpha_bar_t={alpha_bar_t_val:.6e}")
                print(f"  like_scalar_t={like_scalar_val:.6e}")
                print(f"  ||grad_like||_2={grad_like_norm:.6e}")
                print(f"  ||correction||_2={correction_norm:.6e}")
                print(f"  grad_like finite: {torch.isfinite(grad_like).all().item()}, correction finite: {torch.isfinite(correction).all().item()}")
                
                # Hard equivalence test for gamma=0.5
                if self.exp_key == 'D' and abs(self.gamma - 0.5) < 1e-6:
                    # Compute what A would produce
                    like_scalar_A = beta_t.view(1, *shape_ones)
                    correction_A = self.dps_lambda * like_scalar_A * grad_like_H0
                    rel_error = torch.linalg.vector_norm(correction - correction_A, dim=tuple(range(1, correction.ndim))).mean().item()
                    rel_error /= (torch.linalg.vector_norm(correction_A, dim=tuple(range(1, correction_A.ndim))).mean().item() + 1e-12)
                    print(f"  [EQUIVALENCE TEST] ||correction_D - correction_A|| / ||correction_A|| = {rel_error:.6e}")
                    if rel_error > 1e-6:
                        print(f"  WARNING: D(gamma=0.5) should match A but rel_error={rel_error:.6e} > 1e-6")
            
            # Closed-form debug prints (Eprime diagnostic)
            if self.debug_cf and self.exp_key == 'Eprime' and t in (self.dm.num_timesteps - 1, self.dm.num_timesteps // 2 - 1, 0):
                a_val = float(self.dm.alphas_cumprod[t].item())
                sigma2_val = float(self.sigma_y2) if self.sigma_y2 is not None else float('nan')
                g_norm = torch.linalg.vector_norm(grad_like, dim=tuple(range(1, grad_like.ndim))).mean().item()
                upd_norm = torch.linalg.vector_norm(correction, dim=tuple(range(1, correction.ndim))).mean().item()
                print(f"[DEBUG_CF Eprime t={t}] alpha_bar_t={a_val:.6e}, sigma2={sigma2_val:.6e}")
                print(f"  ||score_like_cf||_2={g_norm:.6e}, ||like_update||_2={upd_norm:.6e}")

        # 4) Clip the FINAL likelihood correction (not the gradient): norm cap (like cov path) or elementwise.
        if self.step_clip is not None:
            if self.like_clip_mode == 'norm':
                norm_dims_like = tuple(range(1, correction.ndim))
                dx_norm_like = torch.linalg.vector_norm(correction, dim=norm_dims_like, keepdim=True)
                scale_like = torch.clamp(
                    float(self.step_clip) / (dx_norm_like + 1e-10), max=1.0
                )
                correction = correction * scale_like
            else:
                correction = torch.clamp(correction, -self.step_clip, self.step_clip)

        # 5) Covariance side correction with configurable scaling and normalization
        if self.cov_lambda > 0:
            # If use_t_start_scaling is enabled, compute per-step cov_lambda_eff
            # cov_lambda_eff(t) = cov_lambda_base * sqrt(beta[t])
            if self.use_t_start_scaling:
                eps = 1e-12
                cov_lambda_eff = self.cov_lambda_base * torch.sqrt(beta_t + eps)
            else:
                cov_lambda_eff = self.cov_lambda
            
            # Compute scaling factor zeta_t for covariance guidance.
            # Default behavior: choose based on cov_scale_mode.
            # If cov_beta_power is provided, override with zeta_t = beta_t**p.
            beta_scalar = self.dm.betas[t].to(x_prior.device, dtype=x_prior.dtype)  # scalar tensor
            if self.cov_beta_power is not None:
                eps = 1e-12
                zeta_t = torch.pow(beta_scalar + eps, float(self.cov_beta_power)).view(1, *shape_ones)
            else:
                if self.cov_scale_mode == 'constant':
                    zeta_t = torch.ones(1, *shape_ones, device=x_prior.device, dtype=x_prior.dtype)
                elif self.cov_scale_mode == 'sqrt_beta_t':
                    eps = 1e-12
                    zeta_t = torch.sqrt(beta_scalar + eps).view(1, *shape_ones)
                elif self.cov_scale_mode == 'snr_aware':
                    eps = 1e-12
                    zeta_t = (1.0 / torch.sqrt(beta_scalar + eps)).view(1, *shape_ones)
                else:  # 'beta_t' (default)
                    zeta_t = beta_scalar.view(1, *shape_ones)
            
            # Compute covariance gradient
            grad_cov_raw = compute_cov_grad(x_prior, cov, self.sigma_y2)
            
            # Original version: cov_correction = cov_lambda * beta_t * grad_cov
            # This is the simplest form without normalization or complex scaling
            # Only applies to 'beta_t' mode
            # Backward-compatibility fast path (original implementation):
            # cov_correction = cov_lambda * beta_t * grad_cov_raw with element-wise clamp.
            #
            # If user sets cov_beta_power=1.0, they typically expect it to be identical to
            # cov_scale_mode='beta_t'. We treat p≈1 as equivalent here to preserve that.
            beta_power_is_one = (
                self.cov_beta_power is not None and abs(float(self.cov_beta_power) - 1.0) < 1e-12
            )
            is_beta_t_equivalent = (
                (self.cov_beta_power is None or beta_power_is_one) and
                self.cov_grad_norm == 'none' and
                not self.use_t_start_scaling and
                self.cov_scale_mode == 'beta_t'
            )
            if is_beta_t_equivalent:
                # Original simple version: cov_correction = cov_lambda * beta_t * grad_cov
                zeta_t_for_calc = beta_scalar.view(1, *shape_ones)
                correction_cov_preclip = self.cov_lambda * zeta_t_for_calc * grad_cov_raw
            else:
                # Normalize gradient if requested
                grad_cov = normalize_cov_grad(
                    grad_cov_raw, x_prior, cov, 
                    norm_mode=self.cov_grad_norm, 
                    eps=1e-8
                )
                
                # Apply scaling: dx_cov = cov_lambda * zeta_t * grad
                # Note: compute_cov_grad returns ∇(-||X X^H - cov||_F^2), so we use positive direction
                # to maximize -||X X^H - cov||_F^2 (i.e., minimize ||X X^H - cov||_F^2)
                # If use_t_start_scaling, cov_lambda_eff already includes sqrt(beta_t), so zeta_t should be 1.0
                if self.use_t_start_scaling:
                    # When using per-step scaling, we don't need additional zeta_t scaling
                    # because cov_lambda_eff already includes sqrt(beta_t)
                    correction_cov_preclip = cov_lambda_eff.view(1, *shape_ones) * grad_cov
                else:
                    # Normal mode: use cov_lambda * zeta_t * grad (no negative sign!)
                    correction_cov_preclip = cov_lambda_eff * zeta_t * grad_cov
            
            # Clip covariance correction
            clip_threshold = self.cov_step_clip if self.cov_step_clip is not None else self.step_clip
            if clip_threshold is not None:
                # Decide clipping mode. Default ('auto') preserves backward compatibility:
                # elementwise for legacy beta_t-equivalent path, norm-based otherwise.
                clip_mode = self.cov_clip_mode
                if clip_mode == 'auto':
                    clip_mode = 'elementwise' if is_beta_t_equivalent else 'norm'

                if clip_mode == 'elementwise':
                    correction_cov = torch.clamp(correction_cov_preclip, -clip_threshold, clip_threshold)
                    scale_factor = None  # for diagnostics below
                elif clip_mode == 'norm':
                    norm_dims_clip = tuple(range(1, correction_cov_preclip.ndim))
                    dx_norm = torch.linalg.vector_norm(correction_cov_preclip, dim=norm_dims_clip, keepdim=True)
                    scale_factor = torch.clamp(clip_threshold / (dx_norm + 1e-10), max=1.0)
                    correction_cov = correction_cov_preclip * scale_factor
                else:
                    raise ValueError(f"Unknown cov_clip_mode: {self.cov_clip_mode}. Use 'auto', 'elementwise', or 'norm'.")
                
                # Record diagnostic info if enabled
                if diagnostic_recorder is not None:
                    if clip_mode == 'elementwise':
                        diagnostic_recorder.record_clip_rate(correction_cov_preclip, correction_cov)
                    else:
                        # Norm-based clipping: approximate clip-rate via scaling factor
                        clip_applied = (scale_factor < 1.0).float().mean().item()
                        correction_cov_dummy = correction_cov_preclip * (1.0 - scale_factor) + correction_cov
                        diagnostic_recorder.record_clip_rate(correction_cov_preclip, correction_cov_dummy)
            else:
                correction_cov = correction_cov_preclip
                if diagnostic_recorder is not None:
                    diagnostic_recorder.record_clip_rate(correction_cov, correction_cov)
            
            # Debug logging for cov scaling
            if self.debug_cov_scaling:
                norm_dims = tuple(range(1, grad_cov_raw.ndim))
                grad_cov_raw_norm = torch.linalg.vector_norm(grad_cov_raw, dim=norm_dims).mean().item()
                
                # For original version, grad_cov_normed is the same as grad_cov_raw
                # Only applies to 'beta_t' mode
                beta_power_is_one_dbg = (
                    self.cov_beta_power is not None and abs(float(self.cov_beta_power) - 1.0) < 1e-12
                )
                is_beta_t_equivalent_debug = (
                    (self.cov_beta_power is None or beta_power_is_one_dbg) and
                    self.cov_grad_norm == 'none' and
                    not self.use_t_start_scaling and
                    self.cov_scale_mode == 'beta_t'
                )
                if is_beta_t_equivalent_debug:
                    grad_cov_normed_norm = grad_cov_raw_norm
                else:
                    grad_cov_normed_norm = torch.linalg.vector_norm(grad_cov, dim=norm_dims).mean().item()
                
                dx_cov_preclip_norm = torch.linalg.vector_norm(correction_cov_preclip, dim=norm_dims).mean().item()
                dx_cov_postclip_norm = torch.linalg.vector_norm(correction_cov, dim=norm_dims).mean().item()
                
                # Check if clipping was applied
                if clip_threshold is not None:
                    if is_beta_t_equivalent_debug:
                        # Element-wise comparison for original version
                        diff = (correction_cov_preclip != correction_cov)
                        diff_flat = diff.flatten(start_dim=1)
                        clip_mask = diff_flat.any(dim=1)
                        cov_clip_applied = clip_mask.float().mean().item()
                    else:
                        # Norm-based clipping
                        norm_dims_debug = tuple(range(1, correction_cov_preclip.ndim))
                        dx_norm_pre = torch.linalg.vector_norm(correction_cov_preclip, dim=norm_dims_debug, keepdim=True)
                        dx_norm_post = torch.linalg.vector_norm(correction_cov, dim=norm_dims_debug, keepdim=True)
                        clip_mask = (dx_norm_pre > clip_threshold) & (dx_norm_post < dx_norm_pre - 1e-6)
                        cov_clip_applied = clip_mask.float().mean().item()
                else:
                    cov_clip_applied = 0.0
                
                # zeta_t for original version is just beta_t
                if is_beta_t_equivalent_debug:
                    zeta_t_val = float(beta_scalar.item())
                else:
                    zeta_t_val = float(zeta_t.mean().item()) if isinstance(zeta_t, torch.Tensor) else float(zeta_t)
                
                self.debug_history.append({
                    't': t,
                    'beta_t': float(beta_t.item()),
                    'zeta_t': zeta_t_val,
                    'grad_cov_raw_norm': grad_cov_raw_norm,
                    'grad_cov_normed_norm': grad_cov_normed_norm,
                    'dx_cov_preclip_norm': dx_cov_preclip_norm,
                    'dx_cov_postclip_norm': dx_cov_postclip_norm,
                    'cov_clip_applied': cov_clip_applied,
                })
        else:
            correction_cov = torch.zeros_like(x_prior)

        # Tx covariance off-diagonal regularization:
        # minimize ||off_diag(X^H X)||_F^2  ->  encourages column orthogonality
        if self.tx_cov_lambda > 0:
            with torch.no_grad():
                _beta_tx = self.dm.betas[t].to(x_prior.device, dtype=x_prior.dtype)
                _shape_tx = (1,) * (x_prior.ndim - 1)
                # Reuse cov_scale_mode for consistent time-dependent scaling
                if self.cov_scale_mode == 'sqrt_beta_t':
                    _zeta_tx = torch.sqrt(_beta_tx + 1e-12).view(1, *_shape_tx)
                elif self.cov_scale_mode == 'constant':
                    _zeta_tx = torch.ones(1, *_shape_tx, device=x_prior.device, dtype=x_prior.dtype)
                else:  # 'beta_t' (default) or 'snr_aware' fall back to beta_t
                    _zeta_tx = _beta_tx.view(1, *_shape_tx)

                _grad_tx = compute_tx_cov_grad(x_prior)
                _corr_tx_raw = self.tx_cov_lambda * _zeta_tx * _grad_tx

                # Norm-based clipping (same threshold as cov correction)
                _clip = self.cov_step_clip if self.cov_step_clip is not None else self.step_clip
                if _clip is not None:
                    _nd = tuple(range(1, _corr_tx_raw.ndim))
                    _scale = torch.clamp(
                        _clip / (torch.linalg.vector_norm(_corr_tx_raw, dim=_nd, keepdim=True) + 1e-10),
                        max=1.0,
                    )
                    correction_cov = correction_cov + _corr_tx_raw * _scale
                else:
                    correction_cov = correction_cov + _corr_tx_raw

        # Record diagnostic info if enabled
        if diagnostic_recorder is not None:
            diagnostic_recorder.record_corrections(correction, correction_cov, float(beta_t.item()))

        # 4) Apply DPS correction to the prior estimate
        # For exp F, x_prev is already computed (paper-style); for others, compute as usual.
        if self.exp_key == 'F':
            x_prev = x_prev + correction_cov
        else:
            x_prev = x_prior + correction + correction_cov
        
        # Debug mode: print state change
        if self.debug_likelihood:
            state_change_norm = torch.linalg.vector_norm(x_prev - x_prior, dim=tuple(range(1, x_prev.ndim))).mean().item()
            print(f"  ||x_prev - x_prior||_2={state_change_norm:.6e}")
            print(f"  x_prev finite: {torch.isfinite(x_prev).all().item()}")
            print()
            # Disable debug after 3 steps
            if hasattr(self, '_debug_step_count'):
                self._debug_step_count += 1
                if self._debug_step_count >= 3:
                    self.debug_likelihood = False
            else:
                self._debug_step_count = 1
        
        return x_prev

    @torch.no_grad()
    def reverse_loop_dps(
        self,
        y: Tensor,
        cov: Tensor,
        x_T: Optional[Tensor] = None,
        return_all_timesteps: bool = False,
        num_steps: Optional[int] = None,
        snr: Optional[float] = None,
        obs_snr_db: Optional[float] = None,
        diagnostic_recorder=None,
        t_start_override: Optional[int] = None,
    ):
        """
        Full DPS reverse loop, conditioned on observation y.

        Parameters
        ----------
        y : Tensor
            Observation batch, shape (B, *data_shape_real_or_complex).
        cov: Tensor
            Covariance matrix, shape (B, 2, d_out, d_out).
        x_T : Tensor, optional
            Initial noise sample. If None, sample from N(0, I).
        return_all_timesteps : bool
            If True, returns all intermediate samples with shape (B, T+1, ...).
        num_steps : int, optional
            Number of reverse steps to use. If None, uses self.dm.num_timesteps.
            If num_steps < num_timesteps, uniformly samples timesteps.
        snr : float, optional
            Signal-to-noise ratio of the observation. If provided, starts reverse
            process from the timestep that best matches this SNR (as in equation 7).
            If None, starts from the maximum timestep (T-1).
        t_start_override : int, optional
            If set, use this diffusion index as the first reverse step (clamped to
            ``[0, num_timesteps-1]``), overriding SNR-matched ``t*`` (orthogonal path)
            and ``snr_match['t_start']`` (Gaussian / nonorthogonal pilots). Does not
            change how ``x_T`` is chosen; only the timestep schedule.

        Returns
        -------
        Tensor
            Final posterior samples (or all intermediate samples if requested).
        """
        B = y.shape[0]
        gp_ctx = getattr(self, "_gaussian_pilot_ctx", None)

        if gp_ctx is not None:
            self._like_gate_cached = None
            sm = gp_ctx.get("snr_match") if isinstance(gp_ctx, dict) else None
            if sm is not None:
                # Spatial pilot: SNR match + init (see gaussian_pilot_snr_match; blends to orthonormal as C→I).
                t_start = int(sm["t_start"])
                if x_T is None:
                    x_T = sm["x_init_ang"].to(device=y.device, dtype=y.dtype)
            else:
                # Legacy spatial pilot: full reverse chain from t=T-1, pure noise init.
                t_start = self.dm.num_timesteps - 1
                if x_T is None:
                    x_T = self.dm.noise_multiplier * torch.randn(
                        (B, *self.dm.data_shape),
                        device=self.device,
                        dtype=y.dtype,
                    )
        else:
            # Cache SNR-gate for this run (EXP H only). This SNR is the observation-level SNR, constant per run.
            if self.exp_key == 'H' and self.like_snr_gate:
                if obs_snr_db is None:
                    # If caller doesn't provide it, fall back to snr (linear) if available.
                    if snr is not None and snr > 0:
                        obs_snr_db = 10.0 * math.log10(float(snr))
                if obs_snr_db is not None and self.like_snr_delta_db > 0:
                    x = (float(obs_snr_db) - self.like_snr0_db) / self.like_snr_delta_db
                    self._like_gate_cached = 1.0 / (1.0 + math.exp(-x))
                else:
                    # No valid SNR information; default to no gating.
                    self._like_gate_cached = 1.0

            # Determine starting timestep based on SNR matching (equation 7)
            if snr is not None:
                # Find timestep that best matches the observation SNR
                # t_hat = argmin_l |SNR(Y) - SNR_DM(l)|
                t_start = int(torch.abs(self.dm.snrs - snr).argmin())
            else:
                # Default: start from maximum timestep
                t_start = self.dm.num_timesteps - 1

            if x_T is None:
                # If starting from a specific timestep based on SNR, we should initialize
                # from the observation y (scaled appropriately) rather than pure noise
                if snr is not None:
                    # Normalize input data similar to generate_estimate
                    norm_multiplier = (snr / (1 + snr)) ** 0.5
                    x_T = norm_multiplier * y
                else:
                    # Start from pure noise
                    x_T = self.dm.noise_multiplier * torch.randn(
                        (B, *self.dm.data_shape),
                        device=self.device,
                        dtype=y.dtype,
                    )

        if t_start_override is not None:
            nt = int(self.dm.num_timesteps)
            t0 = int(t_start_override)
            t_start = max(0, min(nt - 1, t0))

        # Starting diffusion index t_start (for logs / CSV); cov scaling still gated by use_t_start_scaling.
        self._last_t_start = int(t_start)
        self._last_beta_t_start = float(self.dm.betas[t_start].item())

        x_t = x_T
        if return_all_timesteps:
            xs = [x_t]

        # Determine which timesteps to use (from t_start down to 0)
        if num_steps is None:
            # Use all timesteps from t_start down to 0
            timesteps = list(reversed(range(t_start + 1)))
        else:
            # Uniformly sample timesteps from t_start down to 0
            if num_steps >= (t_start + 1):
                # If more or equal steps requested than available, use all timesteps
                timesteps = list(reversed(range(t_start + 1)))
            else:
                # Uniformly sample from t_start down to 0
                step_size = t_start / (num_steps - 1) if num_steps > 1 else 0
                timesteps = [int(t_start - i * step_size) for i in range(num_steps)]
                # Ensure we end at 0 and remove duplicates while preserving order
                timesteps = [t for i, t in enumerate(timesteps) if i == 0 or t != timesteps[i-1]]
                if timesteps[-1] != 0:
                    timesteps[-1] = 0
                # Ensure descending order
                timesteps = sorted(set(timesteps), reverse=True)

        # Closed-form debug micro-test: run ONLY 3 steps for fast sanity check (optional)
        if getattr(self, "debug_cf", False) and getattr(self, "debug_cf_microtest", False):
            t_mid = int(t_start // 2)
            timesteps = sorted({t_start, t_mid, 0}, reverse=True)

        # Exp G: paper Algorithm-3 loop runs from tau_S down to tau_1 (not necessarily to 0).
        # If g_tau1 > 0, stop at t=g_tau1 and output x0_hat at the end.
        t_last = 0
        if self.exp_key == 'G' and getattr(self, "g_tau1", 0) > 0:
            tau1 = int(self.g_tau1)
            # Keep only steps >= tau1
            timesteps = [tt for tt in timesteps if tt >= tau1]
            if len(timesteps) == 0 or timesteps[-1] != tau1:
                # Ensure tau1 is included as the final step
                timesteps = sorted(set(timesteps + [tau1]), reverse=True)
            t_last = tau1

        for t in timesteps:
            # Start diagnostic recording for this timestep
            if diagnostic_recorder is not None:
                diagnostic_recorder.start_step(t)
            
            x_t = self.reverse_step_dps(x_t, y, cov, t, diagnostic_recorder=diagnostic_recorder)
            
            # Finish diagnostic recording for this timestep
            if diagnostic_recorder is not None:
                diagnostic_recorder.finish_step()
            
            if return_all_timesteps:
                xs.append(x_t)
            if self.exp_key == 'G' and getattr(self, "g_tau1", 0) > 0:
                t_last = t

        # If Exp G stopped early at tau1>0, output x0_hat computed from x_{tau1}
        if self.exp_key == 'G' and getattr(self, "g_tau1", 0) > 0 and t_last > 0:
            with torch.no_grad():
                batched_times = torch.full((B,), int(t_last), device=self.device, dtype=torch.long)
                eps_hat = self.dm.model(x_t, batched_times)
                a_t = torch.clamp(self.dm.alphas_cumprod[int(t_last)].to(x_t.device), min=1e-12)
                shape_ones = (1,) * (x_t.ndim - 1)
                a_view = a_t.view(1, *shape_ones)
                sqrt_one_minus_a = torch.sqrt(torch.clamp(1.0 - a_view, min=1e-12))
                x0_hat = (x_t - sqrt_one_minus_a * eps_hat) / torch.sqrt(a_view + 1e-12)
                x_t = x0_hat
                if return_all_timesteps:
                    xs.append(x_t)

        # optional clipping (same as DMCE DiffusionModel)
        if getattr(self.dm, "clipping", False):
            x_t = torch.clamp(x_t, -1.0, 1.0)
            if return_all_timesteps:
                xs = [torch.clamp(z, -1.0, 1.0) for z in xs]

        if return_all_timesteps:
            return torch.stack(xs, dim=1)
        return x_t

    @torch.no_grad()
    def generate_posterior_sample(
        self,
        y: Tensor,
        cov: Tensor,
        x_T: Optional[Tensor] = None,
        return_all_timesteps: bool = False,
        num_steps: Optional[int] = None,
        snr: Optional[float] = None,
        obs_snr_db: Optional[float] = None,
        diagnostic_recorder=None,
        tx_cov_lambda: float = 0.0,
        pilot_matrix_X_p: Optional[Tensor] = None,
        y_p_spatial: Optional[Tensor] = None,
        spatial_fft_mode: str = "2D",
        gaussian_snr_match: bool = True,
        gaussian_rho_linear: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
        gaussian_eta_mode: str = "per_sample",
        dataset_avg_trace_over_nt: Optional[float] = None,
        gaussian_snr_match_mode: str = "trace",
        dataset_avg_inv_lambda_min: Optional[float] = None,
        pilot_likelihood_mode: Optional[str] = None,
        pilot_likelihood_domain: str = "spatial",
        t_start_override: Optional[int] = None,
    ):
        """
        Convenience wrapper mirroring DMCE API.
        
        Parameters
        ----------
        num_steps : int, optional
            Number of reverse steps to use. If None, uses self.dm.num_timesteps.
        snr : float, optional
            Signal-to-noise ratio of the observation. If provided, starts reverse
            process from the timestep that best matches this SNR (equation 7).
        pilot_matrix_X_p : Tensor, optional
            Complex pilot matrix X_p of shape (N_T, N_P). When set with ``y_p_spatial``,
            enables the spatial-pilot likelihood branch (Gaussian i.i.d. or nonorthogonal blend).
        y_p_spatial : Tensor, optional
            Spatial-domain pilot observation (B, 2, N_R, N_P) with Y_p = H X_p + N.
        spatial_fft_mode : str
            ``'1D'`` or ``'2D'``; must match the channel FFT convention (same as Tester mode).
        pilot_likelihood_mode : str, optional
            ``'gaussian'`` (default): ``(Y_p - H X_p)X_p^H`` then FFT. ``'nonorthogonal'``: precomputed
            ``Y' = Y_p X_p^H C^{-1}`` with ``G = (Y' - H_0) C`` then FFT (same algebra, explicit ``Y'`` form).
        pilot_likelihood_domain : str
            ``'spatial'`` (default): existing spatial-then-FFT likelihood direction.
            ``'angular_ls'``: form ``Y'`` in spatial domain, FFT to ``Y_tilde``, Tweedie ``H0_tilde`` in angular
            domain, then ``(Y_tilde - H0_tilde) @ (F_tx C F_tx^H)`` with ``C = X_p X_p^H`` (see
            ``angular_ls_mahalanobis_likelihood_grad_angular``).
        gaussian_snr_match : bool, optional
            If True (default), use a **heuristic** scalar ``SNR_eff`` from LS (``\\mathrm{tr}(C^{-1})``)
            to pick ``t^*`` against isotropic ``dm.snrs`` and set ``x_T = H_{LS}^{ang}/\\sqrt{1+\\eta_{eff}^2}``.
            This approximates colored effective noise by one number; not a rigorous statistical match
            (see ``modules/gaussian_pilot_snr_match``).
        gaussian_rho_linear : float, optional
            Linear SNR rho = 10^(SNR_dB/10) for the pilot observation noise (required for snr_match).
        noise_multiplier : float, optional
            DM noise multiplier (must match pilot simulation).
        gaussian_eta_mode : str, optional
            ``per_sample`` (default) or ``dataset_avg`` (use precomputed mean ``tr(inv(gram))/N_t``).
        dataset_avg_trace_over_nt : float, optional
            Precomputed mean trace factor for ``dataset_avg`` mode (``snr_match_mode=trace``).
        gaussian_snr_match_mode : str, optional
            ``trace`` (default): legacy ``\\mathrm{tr}(C^{-1})`` heuristic; ``worst`` uses
            ``\\sigma_{\\mathrm{eff}}^2 = \\sigma^2/\\lambda_{\\min}(C)`` and
            ``\\mathrm{SNR}_{\\mathrm{eff}} = \\rho\\lambda_{\\min}(C)`` (see ``gaussian_pilot_snr_match``).
        dataset_avg_inv_lambda_min : float, optional
            Precomputed ``\\mathbb{E}[1/\\lambda_{\\min}(C)]`` for ``dataset_avg`` + ``worst`` mode.
        t_start_override : int, optional
            Fixed first reverse timestep index; overrides SNR-based ``t*`` (see ``reverse_loop_dps``).
        """
        # Store temporary tx_cov_lambda to be used in this loop
        old_tx_cov = getattr(self, 'tx_cov_lambda', 0.0)
        self.tx_cov_lambda = tx_cov_lambda

        old_gp = getattr(self, "_gaussian_pilot_ctx", None)
        if pilot_matrix_X_p is not None:
            if y_p_spatial is None:
                raise ValueError("y_p_spatial is required when pilot_matrix_X_p is set.")
            plm = pilot_likelihood_mode or "gaussian"
            if plm not in ("gaussian", "nonorthogonal"):
                raise ValueError("pilot_likelihood_mode must be 'gaussian' or 'nonorthogonal'.")
            pld = (pilot_likelihood_domain or "spatial").strip().lower()
            if pld not in ("spatial", "angular_ls"):
                raise ValueError("pilot_likelihood_domain must be 'spatial' or 'angular_ls'.")
            ctx = {
                "X_p": pilot_matrix_X_p,
                "Y_p": y_p_spatial,
                "fft_mode": str(spatial_fft_mode),
                "pilot_likelihood_mode": plm,
                "pilot_likelihood_domain": pld,
            }
            ctx["Y_p_ang"] = ut.complex_1d_fft(
                y_p_spatial, ifft=False, mode=str(spatial_fft_mode), _4d_array=False
            )
            _nt, _np = pilot_matrix_X_p.shape[0], pilot_matrix_X_p.shape[1]
            _i_rect = _pm_pilot.rect_identity_complex_torch(
                _nt, _np, pilot_matrix_X_p.device, pilot_matrix_X_p.dtype
            )
            # Only for legacy gaussian mode when X_p ≈ I: Y_p^ang - x0_hat shortcut.
            ctx["gaussian_likelihood_angular"] = bool(
                plm == "gaussian"
                and _nt == _np
                and torch.max(torch.abs(pilot_matrix_X_p - _i_rect)).item() < 1e-3
                and pld == "spatial"
            )
            from modules.gaussian_pilot_snr_match import least_squares_channel_batch

            Y_c0 = torch.complex(y_p_spatial[:, 0], y_p_spatial[:, 1])
            if plm == "nonorthogonal":
                ctx["Y_prime_c"] = least_squares_channel_batch(Y_c0, pilot_matrix_X_p)
                gram0 = pilot_matrix_X_p @ pilot_matrix_X_p.conj().transpose(-1, -2)
                eye0 = torch.eye(
                    _nt,
                    dtype=gram0.dtype,
                    device=gram0.device,
                )
                ctx["gram_c"] = gram0 + 1e-6 * eye0
            if pld == "angular_ls":
                # Y' = Y_p X_p^H (X_p X_p^H)^{-1} (same LS batch as nonorthogonal ctx).
                Y_prime_c = least_squares_channel_batch(Y_c0, pilot_matrix_X_p)
                Yp_ri = torch.stack((Y_prime_c.real, Y_prime_c.imag), dim=1)
                Y_tilde_ri = ut.complex_1d_fft(
                    Yp_ri, ifft=False, mode=str(spatial_fft_mode), _4d_array=False
                )
                ctx["Y_tilde_c"] = torch.complex(Y_tilde_ri[:, 0], Y_tilde_ri[:, 1])
                gram_c = pilot_matrix_X_p @ pilot_matrix_X_p.conj().transpose(-1, -2)
                eye_c = torch.eye(_nt, dtype=gram_c.dtype, device=gram_c.device)
                C_sp = gram_c + 1e-6 * eye_c
                ctx["Cprec_tilde_c"] = ut.tx_gram_spatial_to_angular(C_sp)
                # Sanity: X_p ≈ I ⇒ C ≈ I ⇒ F C F^H ≈ I (unitary similarity).
                if (
                    _nt == _np
                    and float(torch.max(torch.abs(pilot_matrix_X_p - _i_rect)).item()) < 1e-3
                ):
                    eye_t = torch.eye(_nt, dtype=ctx["Cprec_tilde_c"].dtype, device=ctx["Cprec_tilde_c"].device)
                    fro_rel = (
                        torch.linalg.matrix_norm(ctx["Cprec_tilde_c"] - eye_t, ord="fro")
                        / (float(_nt) ** 0.5 + 1e-12)
                    )
                    if float(fro_rel.real) > 0.05:
                        warnings.warn(
                            "[pilot_likelihood_domain=angular_ls] X_p is near identity but "
                            f"||Cprec_tilde - I||_F / sqrt(N_t) = {float(fro_rel.real):.3e} is not small; "
                            "check pilot / jitter / FFT convention."
                        )
            # Optional: SNR-matched timestep + LS initialization (Gaussian pilots)
            grho = gaussian_rho_linear
            nm = noise_multiplier
            eta_mode = gaussian_eta_mode
            davg = dataset_avg_trace_over_nt
            if gaussian_snr_match and (grho is not None) and (nm is not None):
                from modules.gaussian_pilot_snr_match import build_gaussian_snr_match

                Y_c = torch.complex(y_p_spatial[:, 0], y_p_spatial[:, 1])
                sm = build_gaussian_snr_match(
                    Y_p_c=Y_c,
                    X_p=pilot_matrix_X_p,
                    rho_linear=float(grho),
                    noise_multiplier=float(nm),
                    dm_snrs=self.dm.snrs,
                    spatial_fft_mode=str(spatial_fft_mode),
                    eta_mode=str(eta_mode),
                    dataset_avg_trace_over_nt=davg,
                    snr_match_mode=str(gaussian_snr_match_mode),
                    dataset_avg_inv_lambda_min=dataset_avg_inv_lambda_min,
                )
                ctx["snr_match"] = sm
            self._gaussian_pilot_ctx = ctx
        else:
            self._gaussian_pilot_ctx = None

        try:
            res = self.reverse_loop_dps(
                y,
                cov=cov,
                x_T=x_T,
                return_all_timesteps=return_all_timesteps,
                num_steps=num_steps,
                snr=snr,
                obs_snr_db=obs_snr_db,
                diagnostic_recorder=diagnostic_recorder,
                t_start_override=t_start_override,
            )
        finally:
            self._gaussian_pilot_ctx = old_gp
            self.tx_cov_lambda = old_tx_cov

        return res
