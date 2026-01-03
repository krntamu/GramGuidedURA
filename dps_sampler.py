"""
DPS (Diffusion Posterior Sampling) Sampler for Channel Estimation.

This module implements DPS on top of an unconditional DiffusionModel, allowing
posterior sampling without retraining.

Key Design Decisions:
1. Gradient correction is applied AFTER the prior reverse step (not before)
2. Deterministic DDIM-style steps (add_random=False) are recommended
3. Step size scales with beta_t (forward process noise variance)
4. Optional gradient/step clipping for numerical stability

Theory:
For observation model y = x + n, n ~ N(0, sigma_y^2 I):
- Likelihood gradient: ∇_x log p(y|x) = (y - x) / sigma_y^2
- DPS update: x_prev = x_prior + lambda_dps * beta_t * grad(x_prior, y)
"""

from typing import Callable, Optional, Literal

import torch
from torch import Tensor
from DMCE import utils as dm_utils


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
        cov_scale_mode: Literal['beta_t', 'sqrt_beta_t', 'constant', 'snr_aware'] = 'beta_t',
        cov_beta_power: Optional[float] = None,
        cov_grad_norm: Literal['none', 'by_x', 'by_r', 'global'] = 'none',
        cov_step_clip: Optional[float] = None,
        cov_clip_mode: Literal['auto', 'elementwise', 'norm'] = 'auto',
        add_random: bool = True,
        H=None,
        sigma_y2: Optional[float] = None,
        lambda_dps: Optional[float] = None,
        exp_key: Literal['A', 'B', 'C', 'D', 'E', 'Eprime', 'F', 'G'] = 'A',
        gamma: float = 1.0,
        like_weight: float = 1.0,
        lw_schedule: Literal['const', 'ramp', 'lastk'] = 'const',
        lw_tau: float = 0.95,
        lw_max: float = 8.0,
        lw_end: float = 1.0,
        lw_k: int = 0,
        g_tau1: int = 0,
        like_beta_power: float = 1.0,
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
        # Clip the final correction (not the gradient) to prevent explosion
        # This allows strong mid-SNR updates while preventing high-SNR instability
        self.grad_clip = None        # Do NOT clip gradients (they can be large, that's OK)
        self.step_clip = 2.0         # Clip final correction magnitude (C=1~5 recommended, using 2.0 as default)

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
        """
        # For experiments A and B, we can use no_grad context
        # For experiment C, we need gradients enabled
        if self.exp_key == 'C':
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

                    # 2) Compute likelihood gradient at the prior estimate (not at noisy x_t)
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
                    else:
                        raise ValueError(f"Unknown exp_key: {self.exp_key}. Must be 'A', 'B', 'C', 'D', 'E', 'Eprime', 'F', or 'G'.")

        # 3) Compute likelihood correction (post-add) for A/B/D/Eprime/C.
        # Exp E uses score injection into x_prior above, so correction is already set to 0.
        # Exp F uses paper-style update, so correction is also 0.
        if self.exp_key not in ('E', 'F', 'G'):
            with torch.no_grad():
                beta_t = self.dm.betas[t]  # scalar
                shape_ones = (1,) * (x_prior.ndim - 1)  # broadcast over non-batch dims

                # Optional: use beta_t**p instead of beta_t in post-add operator (A/B/C/D only).
                # This is a lightweight knob for quick operator ablations.
                beta_like = beta_t
                if self.like_beta_power != 1.0 and self.exp_key in ('A', 'B', 'C', 'D'):
                    # beta_t is positive; allow non-integer powers for experimentation.
                    beta_like = torch.pow(beta_t, self.like_beta_power)

                if self.exp_key == 'A':
                    like_scalar_t = beta_like.view(1, *shape_ones)
                elif self.exp_key == 'B':
                    like_scalar_t = beta_like.view(1, *shape_ones)
                elif self.exp_key == 'D':
                    alpha_bar_t = self.dm.alphas_cumprod[t]
                    alpha_eff = torch.clamp(alpha_bar_t, min=1e-12)
                    exponent = self.gamma - 0.5
                    like_scalar_t = beta_like * (alpha_eff ** exponent)
                    like_scalar_t = like_scalar_t.view(1, *shape_ones).to(x_prior.device)
                elif self.exp_key == 'Eprime':
                    # Diagnostic: sigma_t^2 scaling instead of beta_t
                    a = torch.clamp(self.dm.alphas_cumprod[t].to(x_prior.device), min=1e-12)
                    sigma_t2 = ((1.0 - a) / a).view(1, *shape_ones).to(x_prior.device)
                    like_scalar_t = sigma_t2
                else:
                    # Experiment C: use beta_t (autograd already computed full gradient)
                    like_scalar_t = beta_like.view(1, *shape_ones)

                correction = self.dps_lambda * like_scalar_t * grad_like
            
            # Debug mode: print detailed information for micro-test
            if self.debug_likelihood:
                alpha_bar_t_val = self.dm.alphas_cumprod[t].item() if self.exp_key in ['B', 'D'] else None
                like_scalar_val = like_scalar_t.mean().item() if isinstance(like_scalar_t, torch.Tensor) else like_scalar_t
                grad_like_norm = torch.linalg.vector_norm(grad_like, dim=tuple(range(1, grad_like.ndim))).mean().item()
                correction_norm = torch.linalg.vector_norm(correction, dim=tuple(range(1, correction.ndim))).mean().item()
                
                print(f"[DEBUG t={t}] exp_key={self.exp_key}, gamma={self.gamma if self.exp_key == 'D' else 'N/A'}")
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

        # 4) Clip the FINAL correction (not the gradient) to prevent explosion
        # This allows strong mid-SNR updates while preventing high-SNR instability
        # Using C=2.0 as default (can be tuned between 1-5)
        if self.step_clip is not None:
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
        diagnostic_recorder=None,
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

        Returns
        -------
        Tensor
            Final posterior samples (or all intermediate samples if requested).
        """
        B = y.shape[0]

        # Determine starting timestep based on SNR matching (equation 7)
        if snr is not None:
            # Find timestep that best matches the observation SNR
            # t_hat = argmin_l |SNR(Y) - SNR_DM(l)|
            t_start = int(torch.abs(self.dm.snrs - snr).argmin())
        else:
            # Default: start from maximum timestep
            t_start = self.dm.num_timesteps - 1
        
        # Store t_start info for logging (but don't apply scaling here)
        # Scaling will be applied per-step in reverse_step_dps
        if self.use_t_start_scaling:
            self._last_t_start = t_start
            self._last_beta_t_start = self.dm.betas[t_start].item()

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
        diagnostic_recorder=None,
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
        """
        return self.reverse_loop_dps(
            y,
            cov=cov,
            x_T=x_T,
            return_all_timesteps=return_all_timesteps,
            num_steps=num_steps,
            snr=snr,
            diagnostic_recorder=diagnostic_recorder,
        )
