import math
import os
import os.path as path
import warnings
from functools import partial
from typing import Tuple, Union, Dict

import time
import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from torch import nn
from torch.utils.data import DataLoader
from modules import utils as ut
from tqdm.auto import tqdm
import torch.nn.functional as F

from DMCE import utils, networks, functional


class FlowModel(nn.Module):
    def __init__(self,
                 model: networks.CNN,
                 *,
                 data_shape: Union[Tuple, list],
                 complex_data: bool = True,
                 loss_type: str = 'l2',
                 num_timesteps: int = 100,
                 sigma_min: float = 0.01,
                 sigma_max: float = 50.0,
                 rho: float = 7.0,
                 noise_std: float = 1.0,
                 sampling_eps: float = 0.002,
                 clipping: bool = False,
                 device: Union[str, torch.device] = 'cuda'):
        """
        PyTorch Module that implements the Flow Model and all its functionalities.
        Flow models learn a continuous transformation from noise to data using ODEs.
        This implementation is based on the EDM (Elucidating the Design Space of
        Diffusion-Based Generative Models) framework.

        Parameters
        ----------
        model : PyTorch Module
            Implements the actual Neural Network. It requires a 'forward()' method.
        data_shape : Tuple or list of shape [channels, dim1, dim2, ...]
            Shape of the data the Flow model is working on.
        complex_data : bool
            Specifies, whether the original data type is complex or real
        loss_type : str {'l1', 'l2'}
            Defines which PyTorch loss function to use.
        num_timesteps : int
            Number of timesteps for discretization during training
        sigma_min : float
            Minimum noise level
        sigma_max : float
            Maximum noise level
        rho : float
            Controls the noise schedule curvature
        noise_std : float
            Standard deviation of the noise
        sampling_eps : float
            Small constant for numerical stability during sampling
        clipping : bool
            Specifies, whether the data is clipped to [-1, 1] after generation
        device : str or torch.device
            Device to run the model on
        """
        super().__init__()
        self.device = utils.set_device(device)
        self.num_timesteps = num_timesteps
        self.data_shape = tuple(data_shape)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.noise_std = noise_std
        self.sampling_eps = sampling_eps

        # for complex data, we have to multiply the real and imaginary normal noise parts with 1/sqrt(2)
        self.noise_multiplier = 1 / (2 ** 0.5) if complex_data else 1.0
        self.model = model.to(self.device)
        self.clipping = clipping

        if loss_type not in ['l1', 'l2']:
            raise ValueError(f"Invalid loss type '{loss_type}'.")
        self.loss_type = loss_type

        # Precompute noise schedule
        self._setup_noise_schedule()
        self.num_parameters = utils.count_params(self, only_trainable=True)

    def _setup_noise_schedule(self):
        # tgrid is in [0,1], but we store a table of length num_timesteps
        tgrid  = torch.linspace(0., 1., self.num_timesteps, device=self.device)
        smin_r = self.sigma_min ** (1.0 / self.rho)
        smax_r = self.sigma_max ** (1.0 / self.rho)
        sigma  = (smin_r + tgrid * (smax_r - smin_r)) ** self.rho
        ds_dt  = self.rho * (smin_r + tgrid * (smax_r - smin_r)) ** (self.rho - 1.0) * (smax_r - smin_r)

        self.register_buffer('sigma', sigma)      # length = num_timesteps
        self.register_buffer('sigma_deriv', ds_dt)


    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise level at time t using linear interpolation
        t should be in [0, 1] (normalized continuous time)
        """
        # Ensure t is in [0, 1]
        t_normalized = torch.clamp(t, 0.0, 1.0)

        # Scale to sigma array indices [0, num_timesteps-1]
        indices = t_normalized * (self.num_timesteps - 1)

        # Get integer indices for interpolation
        indices_low = torch.floor(indices).long()
        indices_high = torch.minimum(indices_low + 1, torch.tensor(self.num_timesteps - 1, device=t.device))

        # Get weights for interpolation
        weights_high = indices - indices_low.float()
        weights_low = 1.0 - weights_high

        # Extract sigma values
        sigma_low = self.sigma[indices_low]
        sigma_high = self.sigma[indices_high]

        # Linear interpolation
        sigma_interp = weights_low * sigma_low + weights_high * sigma_high
        return sigma_interp.view(-1, *([1] * (t.dim() - 1)))

    def sigma_to_t(self, sigma_val: torch.Tensor) -> torch.Tensor:
        """Convert sigma value to normalized time t in [0, 1]
        Returns normalized continuous time t ∈ [0, 1]
        """
        # self.sigma: shape [num_timesteps], 单调递增（从 sigma_min 到 sigma_max）
        sigma = self.sigma  # [T], 单调递增
        # 确保 sigma_val 在有效范围内
        s = sigma_val.float() if isinstance(sigma_val, torch.Tensor) else torch.tensor(sigma_val, dtype=torch.float32, device=self.device)
        s = s.clamp(min=sigma.min().item(), max=sigma.max().item())
        
        # 对于标量，扩展为1D tensor
        if s.dim() == 0:
            s = s.unsqueeze(0)
        
        # 使用 searchsorted 在单调递增的数组上查找
        # searchsorted 返回插入位置，即第一个 >= s 的位置
        idx_high = torch.searchsorted(sigma, s, right=False)
        idx_high = idx_high.clamp(min=1, max=self.num_timesteps - 1)  # 确保有 idx_low
        idx_low = idx_high - 1
        
        # 提取对应的 sigma 值
        s_low = sigma[idx_low]
        s_high = sigma[idx_high]
        
        # 线性插值权重
        denom = s_high - s_low
        w = torch.where(denom > 1e-8, (s - s_low) / denom, torch.zeros_like(s))
        
        # 转换为归一化时间 [0, 1]
        t_index = idx_low.float() + w  # 离散索引 [0, num_timesteps-1]
        t_normalized = t_index / (self.num_timesteps - 1)  # 归一化到 [0, 1]
        t_normalized = torch.clamp(t_normalized, 0.0, 1.0)
        
        # 如果输入是标量，返回标量
        if sigma_val.dim() == 0 or (isinstance(sigma_val, float) and not isinstance(sigma_val, torch.Tensor)):
            return t_normalized.item() if t_normalized.numel() == 1 else t_normalized
        return t_normalized


    def get_sigma_deriv(self, t: torch.Tensor) -> torch.Tensor:
        """Get derivative of noise level at time t using linear interpolation
        t should be in [0, 1] (normalized continuous time)
        """
        # Ensure t is in [0, 1]
        t_normalized = torch.clamp(t, 0.0, 1.0)

        # Scale to sigma_deriv array indices [0, num_timesteps-1]
        indices = t_normalized * (self.num_timesteps - 1)

        # Get integer indices for interpolation
        indices_low = torch.floor(indices).long()
        indices_high = torch.minimum(indices_low + 1, torch.tensor(self.num_timesteps - 1, device=t.device))

        # Get weights for interpolation
        weights_high = indices - indices_low.float()
        weights_low = 1.0 - weights_high

        # Extract sigma_deriv values
        sigma_deriv_low = self.sigma_deriv[indices_low]
        sigma_deriv_high = self.sigma_deriv[indices_high]

        # Linear interpolation
        sigma_deriv_interp = weights_low * sigma_deriv_low + weights_high * sigma_deriv_high
        return sigma_deriv_interp.view(-1, *([1] * (t.dim() - 1)))

    def forward_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward process: add noise to data according to the flow schedule

        Parameters
        ----------
        x_0 : Tensor of shape [batch_size, ...]
            Ground truth data samples
        t : Tensor of shape [batch_size]
            Time steps
        noise : optional Tensor of shape [batch_size, ...]
            Noise to add (if None, random noise is sampled)

        Returns
        -------
        x_t : Tensor of shape [batch_size, ...]
            Noisy data at time t
        """
        sigma_t = self.get_sigma(t)
        noise = utils.default(noise, lambda: self.noise_multiplier * torch.randn_like(x_0))
        # Add noise according to the flow schedule
        x_t = x_0 + sigma_t.view(-1, *([1] * (x_0.dim() - 1))) * noise
        return x_t

    def predict_denoised(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the denoised data from noisy data at time t

        Parameters
        ----------
        x_t : Tensor of shape [batch_size, ...]
            Noisy data at time t
        t : Tensor of shape [batch_size]
            Time steps

        Returns
        -------
        x_0_pred : Tensor of shape [batch_size, ...]
            Predicted denoised data
        """
        # The model predicts the denoised data directly
        return self.model(x_t, t)

    def predict_score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the score function (gradient of log probability)

        Returns
        -------
        score : Tensor of shape [batch_size, ...]
            Predicted score function
        """
        x_0_pred = self.predict_denoised(x_t, t)
        sigma_t = self.get_sigma(t)
        # Score = -(x_t - x_0) / sigma_t^2
        score = -(x_t - x_0_pred) / (sigma_t.view(-1, *([1] * (x_t.dim() - 1))) ** 2)
        return score

    def _dxdt(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma_t      = self.get_sigma(t)
        sigma_dt     = self.get_sigma_deriv(t)
        x0_hat       = self.predict_denoised(x_t, t)
        # 计算 score，添加数值稳定性保护
        sigma_t_sq = sigma_t.view(-1, *([1] * (x_t.dim() - 1))) ** 2
        score        = -(x_t - x0_hat) / (sigma_t_sq + 1e-8)
        # 计算 drift
        sigma_dt_view = sigma_dt.view(-1, *([1] * (x_t.dim() - 1)))
        sigma_t_view = sigma_t.view(-1, *([1] * (x_t.dim() - 1)))
        drift        = -sigma_dt_view * sigma_t_view * score
        # 检查 NaN 和 Inf
        drift = torch.where(torch.isfinite(drift), drift, torch.zeros_like(drift))
        return drift

    @torch.no_grad()
    def reverse_step(self, x_t: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Single reverse step using Heun (predictor-corrector) method
        t should be in [0, 1] (normalized continuous time)
        """
        # Heun (predictor-corrector)
        k1 = self._dxdt(x_t, t)
        x_pred = x_t - dt * k1
        t_next = t - dt
        # 确保 t_next 在有效范围内 [0, 1]
        t_next = torch.clamp(t_next, min=0.0, max=1.0)
        k2 = self._dxdt(x_pred, t_next)
        result = x_t - 0.5 * dt * (k1 + k2)
        # 检查 NaN 和 Inf
        result = torch.where(torch.isfinite(result), result, x_t)  # 如果出现 NaN，返回原值
        return result

    # @torch.no_grad()
    # def reverse_step(self, x_t: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
    #     """
    #     Single reverse step using the ODE solver
    #     """
    #     sigma_t = self.get_sigma(t)
    #     sigma_deriv_t = self.get_sigma_deriv(t)

    #     # Predict denoised data
    #     x_0_pred = self.predict_denoised(x_t, t)

    #     # ODE step: dx/dt = -sigma'(t) * sigma(t) * score
    #     score = -(x_t - x_0_pred) / (sigma_t.view(-1, *([1] * (x_t.dim() - 1))) ** 2)
    #     dx_dt = -sigma_deriv_t.view(-1, *([1] * (x_t.dim() - 1))) * sigma_t.view(-1, *([1] * (x_t.dim() - 1))) * score

    #     # Euler step
    #     x_t_minus_dt = x_t - dt * dx_dt
    #     return x_t_minus_dt

    @torch.no_grad()
    def reverse_sample_loop(self,
                            x_t: torch.Tensor,
                            t_start: float,
                            t_end: float = None,
                            *,
                            return_all_timesteps: bool = False,
                            num_steps: int = None) -> torch.Tensor:
        """
        Reverse sampling loop using ODE solver
        t_start and t_end should be in [0, 1] (normalized continuous time)
        """
        if t_end is None:
            t_end = 0.0  # normalized time: final time at the smallest sigma
        if num_steps is None:
            num_steps = self.num_timesteps

        # Ensure t_start and t_end are in [0, 1]
        t_start = max(0.0, min(1.0, float(t_start)))
        t_end = max(0.0, min(1.0, float(t_end)))

        dt = (t_start - t_end) / max(1, num_steps)
        # 确保 dt 不会太小或太大
        if dt < 1e-6:
            dt = 1e-6
        if dt > 1.0:  # dt should be in [0, 1] since t is normalized
            dt = 1.0
            
        t_current = t_start
        x_current = x_t
        x_all = [x_current] if return_all_timesteps else None

        for step_idx in range(num_steps):
            # 确保 t_current 在有效范围内 [0, 1]
            t_current = max(0.0, min(1.0, t_current))
            t_tensor = torch.full((x_current.shape[0],), t_current, device=self.device)
            x_current = self.reverse_step(x_current, t_tensor, dt)
            
            # 检查 NaN 和 Inf
            if torch.any(~torch.isfinite(x_current)):
                print(f"Warning: NaN/Inf detected at step {step_idx}, t={t_current:.4f}")
                x_current = torch.where(torch.isfinite(x_current), x_current, torch.zeros_like(x_current))
            
            t_current -= dt
            if return_all_timesteps:
                x_all.append(x_current)

        # Clip if needed
        if self.clipping:
            x_current = torch.clamp(x_current, -1, 1)
            if return_all_timesteps:
                x_all = [torch.clamp(x, -1, 1) for x in x_all]

        if return_all_timesteps:
            return torch.stack(x_all, dim=1)
        else:
            return x_current

    @torch.no_grad()
    def generate_new_samples(self, n_samples: int, *, noise: torch.Tensor = None,
                            return_all_timesteps: bool = False) -> torch.Tensor:
        """
        Generate new samples from noise
        Uses normalized time t ∈ [0, 1]
        """
        # Start from pure noise at maximum noise level
        x_t = utils.default(
            noise,
            lambda: self.noise_multiplier * torch.randn((n_samples, *self.data_shape), device=self.device)
        ) * self.sigma_max  # single shot scaled noise
        # Reverse sample from t_start = 1.0 (normalized time, corresponds to sigma_max)
        x_0 = self.reverse_sample_loop(x_t, t_start=1.0, return_all_timesteps=return_all_timesteps)
        return x_0


    @torch.no_grad()
    def generate_estimate(self, y: torch.Tensor, snr: float, *, return_all_timesteps=False, num_steps: int = None):
        """
        Generate estimate from noisy data
        Uses normalized time t ∈ [0, 1]
        """
        # 线性 SNR → 观测噪声 std（匹配 awgn）
        # awgn 函数中噪声是: (1/sqrt(SNR)) * multiplier * randn
        # 所以观测噪声的标准差是: (1/sqrt(SNR)) * multiplier
        sigma_y = (1.0 / (snr ** 0.5)) * self.noise_multiplier
        
        # 确保 sigma_y 在有效范围内
        sigma_y = max(self.sigma_min, min(self.sigma_max, sigma_y))
        
        # 把观测直接作为初值
        x_t = y
        
        # 由 σ_y 反查 t_start（归一化时间 [0, 1]）
        t_start_val = self.sigma_to_t(sigma_y)
        if isinstance(t_start_val, torch.Tensor):
            t_start = t_start_val.item() if t_start_val.numel() == 1 else float(t_start_val)
        else:
            t_start = float(t_start_val)
        
        # 确保 t_start 在有效范围内 [0, 1]
        t_start = max(0.0, min(1.0, t_start))
        
        # 逆向积分，使用指定的 num_steps（默认使用模型初始化时的 num_timesteps）
        return self.reverse_sample_loop(x_t, t_start, return_all_timesteps=return_all_timesteps, num_steps=num_steps)

    @torch.no_grad()
    def generate_conditional_estimate(self,
                                      y: torch.Tensor,
                                      snr: float,
                                      *,
                                      n_samples: int = 10,
                                      n_steps: int = 100,
                                      return_all_timesteps: bool = False,
                                      guidance_scale: float = None,
                                      adaptive_guidance: bool = True) -> torch.Tensor:
        """
        Generate conditional estimate using guided sampling (based on Y=X+N)
        """
        # Convert SNR to noise level
        noise_level = 1.0 / (snr ** 0.5)

        # Adaptive guidance scale calculation
        if adaptive_guidance:
            if guidance_scale is None:
                # Method 1: SNR-based adaptive scaling
                guidance_scale = self._compute_adaptive_guidance_scale(snr, noise_level)
        else:
            # Use fixed scale if provided, otherwise default
            guidance_scale = guidance_scale if guidance_scale is not None else 0.05

        # Initialize X from noise
        batch_size = y.shape[0]
        x_current = torch.randn(n_samples, batch_size, *self.data_shape, device=self.device)
        # Use normalized time [0, 1]
        dt = 1.0 / n_steps
        x_all = [x_current] if return_all_timesteps else None

        for i in range(n_steps):
            # t in [0, 1] (normalized continuous time)
            t = torch.ones(n_samples, batch_size, device=self.device) * (i * dt)

            # Reshape x_current for model prediction: [n_samples * batch_size, ...]
            x_flat = x_current.view(-1, *self.data_shape)
            t_flat = t.view(-1)

            # Get velocity from model (predict denoised)
            x_0_pred_flat = self.predict_denoised(x_flat, t_flat)
            x_0_pred = x_0_pred_flat.view(n_samples, batch_size, *self.data_shape)
            velocity = (x_0_pred - x_current) / dt  # Approximate velocity

            # Update X using ODE step
            x_current = x_current + velocity * dt

            # Add conditional guidance: score_y_given_xt = (Y - X) / sigma^2 (H=I case)
            # Compute difference Y - X
            diff = y.unsqueeze(0).expand(n_samples, -1, -1, -1, -1) - x_current  # [n_samples, batch_size, C, H, W]

            # Compute conditional score
            score_y_given_xt = diff / (noise_level ** 2)

            # Normalize score for stability
            norm = torch.norm(diff, p=2, dim=(2, 3, 4), keepdim=True) + 1e-8
            score_y_given_xt = score_y_given_xt * (guidance_scale if guidance_scale is not None else 0.05) / norm

            # Apply conditional guidance
            x_current = x_current + score_y_given_xt

            if return_all_timesteps:
                x_all.append(x_current)

        if return_all_timesteps:
            return torch.stack(x_all, dim=1)
        else:
            return x_current

    def _compute_adaptive_guidance_scale(self, snr: float, noise_level: float) -> float:
        """
        Compute adaptive guidance scale based on SNR and noise level
        """
        # Method 1: SNR-based scaling (logarithmic)
        # Higher SNR -> smaller guidance scale (more confident in observation)
        # Lower SNR -> larger guidance scale (less confident in observation)
        guidance_scale = 0.1 * (1.0 + 1.0 / (1.0 + snr))

        # Clamp to reasonable range
        guidance_scale = max(0.01, min(0.5, guidance_scale))
        return guidance_scale

    def _compute_dynamic_guidance_scale(self, snr: float, noise_level: float, step: int, total_steps: int) -> float:
        """
        Compute dynamic guidance scale that changes during sampling
        """
        # Base scale
        base_scale = self._compute_adaptive_guidance_scale(snr, noise_level)

        # Time-dependent scaling
        progress = step / total_steps
        # Method 3: Bell curve (strong in middle, weak at start/end)
        guidance_scale = base_scale * (1.0 - 0.5 * (4 * progress * (1 - progress)))
        return guidance_scale

    @property
    def loss_fn(self):
        """Returns the correct PyTorch loss function"""
        if self.loss_type == 'l1':
            return nn.functional.l1_loss
        elif self.loss_type == 'l2':
            return nn.functional.mse_loss
        else:
            raise ValueError(f"Invalid loss type '{self.loss_type}'.")

    def forward(self, x_0: torch.Tensor, noise: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for training
        t should be in [0, 1] (normalized continuous time)
        """
        b = x_0.shape[0]

        # Sample random time steps in [0, 1]
        t = utils.default(t, lambda: torch.rand(b, device=self.device))

        # Forward process: add noise
        x_t = self.forward_sample(x_0, t, noise)

        # Predict denoised data
        x_0_pred = self.predict_denoised(x_t, t)

        # Compute loss
        loss = self.loss_fn(x_0_pred, x_0)
        return loss


class FlowTrainer(object):
    def __init__(self,
                 model: FlowModel,
                 data_train: torch.Tensor,
                 data_val: torch.Tensor,
                 *,
                 batch_size: int = 128,
                 lr_init: float = 1e-3,
                 lr_step_multiplier: float = 0.5,
                 epochs_until_lr_step: int = 150,
                 num_epochs: int = 500,
                 val_every_n_batches: int = None,
                 mode: str = '1D',
                 track_fid_score: bool = False,
                 track_val_loss: bool = True,
                 track_mmd: bool = False,
                 use_fixed_gen_noise: bool = True,
                 save_mode: str = 'best',
                 dir_result: str = '../results',
                 use_ray: bool = False,
                 complex_data: bool = True,
                 num_min_epochs: int = 1,
                 num_epochs_no_improve: int = 1,
                 fft_pre: bool = False):
        """Trainer class for a FlowModel instance."""
        self.model = model
        self.device = model.device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.val_every_n_batches = val_every_n_batches
        self.mode = mode
        self.track_fid_score = track_fid_score
        self.track_val_loss = track_val_loss
        self.track_mmd = track_mmd
        self.use_fixed_gen_noise = use_fixed_gen_noise
        self.save_mode = save_mode
        self.dir_result = dir_result
        os.makedirs(self.dir_result, exist_ok=True)
        self.use_ray = use_ray
        self.epoch = 0
        self.checkpoint = 0
        self.complex_data = complex_data
        self.num_min_epochs = num_min_epochs
        self.num_min_epochs_no_improve = num_epochs_no_improve
        self.fft_pre = fft_pre

        # instantiate optimizer and lr scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=epochs_until_lr_step,
            gamma=lr_step_multiplier,
            verbose=True
        )

        # training data preparation
        self.num_samples, *_ = data_train.shape
        if fft_pre:
            data_train = ut.complex_1d_fft(data_train, ifft=False, mode=mode)
        data_train = data_train.to(dtype=torch.float32)
        self.dataloader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.num_batches = len(self.dataloader)

        # validation data preparation
        self.num_val_samples, *_ = data_val.shape
        if track_val_loss:
            if fft_pre:
                data_val = ut.complex_1d_fft(data_val, ifft=False, mode=self.mode)
            self.data_val = data_val.to(dtype=torch.float32)
            self.noise_val = self.model.noise_multiplier * torch.randn_like(self.data_val)
            # Sample time in [0, 1] (normalized continuous time)
            self.t_val = torch.rand(self.num_val_samples, device=self.device)

        # prepare everything required for FID calculation
        self.num_gen_samples = 0
        self.num_fid_samples = 0
        if track_fid_score:
            if mode == '1D' and self.complex_data:
                self.num_fid_samples = min(1000, self.num_val_samples)
                feature_func = partial(utils.real2cmplx, dim=1, squeezed=True)
            elif mode == '1D' and not self.complex_data:
                self.num_fid_samples = min(1000, self.num_val_samples)
                feature_func = np.squeeze
            elif mode == '2D':
                self.num_fid_samples = min(100, self.num_val_samples)
                inception = InceptionV3(normalize_input=False, requires_grad=False)
                inception.to(device=self.device)
                inception.eval()
                feature_func = partial(functional.feature_func2d, inception=inception)
            else:
                raise ValueError(f'Data mode {self.mode} is not supported.')

            self.generation_metric = partial(functional.compute_fid_score, feature_func=feature_func)
            self.num_gen_samples = self.num_fid_samples

            # data used for FID calculation is a random subset of the validation data
            self.data_fid = utils.get_random_subset(self.data_val, num_samples=self.num_fid_samples)

        # prepare everything required for MMD calculation
        self.num_mmd_samples = 0
        if track_mmd:
            if mode == '1D':
                self.num_mmd_samples = min(2000, self.num_val_samples)
            elif mode == '2D':
                self.num_mmd_samples = min(2000, self.num_val_samples)
            else:
                raise ValueError(f'Data mode {self.mode} is not supported.')
            self.num_gen_samples = max(self.num_gen_samples, self.num_mmd_samples)

            # data used for MMD calculation is a random subset of the validation data
            self.data_mmd = utils.get_random_subset(self.data_val, num_samples=self.num_mmd_samples).cpu()

        # sample starting noise if new data samples should be generated from same noise in each validation iteration
        if (track_fid_score or track_mmd) and self.use_fixed_gen_noise:
            self.gen_noise = self.model.noise_multiplier * torch.randn_like(
                self.data_mmd if self.num_mmd_samples >= self.num_fid_samples else self.data_fid
            )
            self.gen_noise = self.gen_noise.to(dtype=torch.float32)

    def get_checkpoint_dict(self, **metrics: dict) -> dict:
        """Construct a dictionary with all the information regarding the training procedure that should be stored."""
        checkpoint_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'batch_size': self.batch_size,
            'mode': self.mode,
            'dir_result': self.dir_result,
            'checkpoint': self.checkpoint,
        }
        checkpoint_dict.update(**metrics)
        return checkpoint_dict

    def save_model(self, **metrics: dict):
        """Save the training state in a .pt file."""
        new_dict = self.get_checkpoint_dict(**metrics)

        # generate path to the new model file
        dir_model = path.join(self.dir_result, 'train_models')
        os.makedirs(dir_model, exist_ok=True)
        filepath = path.join(dir_model, f'model-{self.checkpoint}.pt')

        if self.save_mode == 'all':
            torch.save(new_dict, filepath)

        elif self.save_mode == 'best':
            old_files = os.listdir(dir_model)
            if not old_files:
                torch.save(new_dict, filepath)
            else:
                try:
                    save_new = True
                    for old_file in old_files:
                        old_dict = torch.load(path.join(dir_model, str(old_file)), map_location=self.device)
                        if new_dict['val_loss'] > old_dict['val_loss']:
                            save_new = False
                            break
                    if save_new:
                        torch.save(new_dict, filepath)
                        for old_file in old_files:
                            os.remove(path.join(dir_model, str(old_file)))
                except OSError as error:
                    warnings.warn(f'\n{error}\nFalling back to save_mode = \'all\'!')
                    self.save_mode = 'all'

        elif self.save_mode == 'newest':
            old_files = os.listdir(dir_model)
            if not old_files:
                torch.save(new_dict, filepath)
            else:
                try:
                    torch.save(new_dict, filepath)
                    for file in old_files:
                        os.remove(path.join(dir_model, str(file)))
                except OSError as error:
                    warnings.warn(f'\n{error}\nFalling back to save_mode = \'all\'!')
                    self.save_mode = 'all'
        else:
            raise NotImplementedError(self.save_mode)

    def load_model(self, checkpoint: int = None, filepath: str = None):
        """Loads parameters and whole models from a .pt file."""
        if not utils.exists(checkpoint) and not utils.exists(filepath):
            raise ValueError('Either checkpoint or filepath required for model loading')

        filepath = utils.default(filepath, path.join(self.dir_result, 'train_models', f'model-{checkpoint}.pt'))
        if not path.isfile(filepath):
            raise ValueError('Model file does not exist.')

        load_dict = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(load_dict['model'])
        self.optimizer.load_state_dict(load_dict['optimizer'])
        self.lr_scheduler.load_state_dict(load_dict['lr_scheduler'])
        self.epoch = load_dict['epoch']
        self.batch_size = load_dict['batch_size']
        self.mode = load_dict['mode']
        self.dir_result = load_dict['dir_result']
        self.checkpoint = load_dict['checkpoint']
        self.model.to(device=self.device)

    def validate(self, loss: Union[float, torch.Tensor]) -> Tuple[float, float, float]:
        """Validation branch of the training procedure."""
        self.model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            val_loss = loss
            if self.track_val_loss:
                val_loss = self.model(self.data_val.to(device=self.device),
                                      self.noise_val.to(device=self.device),
                                      self.t_val.to(device=self.device))
                val_loss = float(val_loss)

            if self.num_gen_samples != 0:
                data_sampled = []
                diff = self.num_gen_samples
                idx = 0
                while diff > 0:
                    n_samples = min(diff, 512)
                    if self.use_fixed_gen_noise:
                        gen_noise = self.gen_noise[idx:idx + n_samples].to(device=self.device)
                    else:
                        gen_noise = None
                    data_sampled.append(self.model.generate_new_samples(
                        n_samples=n_samples, noise=gen_noise, return_all_timesteps=False))
                    diff -= n_samples
                    idx += n_samples
                data_sampled = torch.cat(data_sampled, dim=0)

            fid_score = None
            if self.track_fid_score:
                fid_score = self.generation_metric(
                    self.data_fid.to(device=self.device),
                    data_sampled[:self.num_fid_samples]
                )
                fid_score = float(fid_score)

            mmd = None
            if self.track_mmd:
                with utils.set_num_threads_context(num_threads=int(os.cpu_count() // 2)):
                    mmd = functional.calculate_mmd(
                        self.data_mmd.cpu(), data_sampled[:self.num_mmd_samples].cpu())
                mmd = float(mmd)

            # Construct metrics dictionary and save the validation results
            metrics = {}
            metrics.update({'val_loss': val_loss}) if self.track_val_loss else None
            metrics.update({'fid_score': fid_score}) if self.track_fid_score else None
            metrics.update({'mmd': mmd}) if self.track_mmd else None

            self.save_model(**metrics)
            self.print_validation_msg(**metrics)

        self.model.train()
        torch.cuda.empty_cache()
        return val_loss, fid_score, mmd

    def print_validation_msg(self, **metrics):
        """prints a message to the console with information about the training progress"""
        msg = f'Epoch {self.epoch}/{self.num_epochs + 1}:'
        for key in metrics.keys():
            msg += f' {key} = {metrics[key]} |'
        msg = msg[:-1]
        print(msg, flush=True)

    def train(self) -> Dict:
        """Main training loop for the FlowModel instance."""
        curr_num_batches = 0

        # Initial validation
        fid_scores = []
        val_losses = []
        train_losses = []
        mmds = []

        loss = 0
        val_loss, fid_score, mmd = self.validate(loss)
        val_losses.append(val_loss)
        train_losses.append(val_loss)
        fid_scores.append(fid_score)
        mmds.append(mmd)
        self.checkpoint += 1
        self.epoch += 1

        early_stopping = Early_stopping(min_epochs=self.num_min_epochs,
                                        num_epochs_no_improve=self.num_min_epochs_no_improve)

        # Training loop
        while self.epoch <= self.num_epochs:
            train_losses_epochs = []
            for batch, data_batch in enumerate(self.dataloader if not self.use_ray else self.dataloader):
                loss = self.model(data_batch.to(device=self.device))
                train_losses_epochs.append(float(loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                curr_num_batches += 1

            train_losses.append(np.mean(train_losses_epochs))

            # validation branch
            with torch.no_grad():
                val_loss, fid_score, mmd = self.validate(loss)
            val_losses.append(val_loss)
            fid_scores.append(fid_score)
            mmds.append(mmd)
            self.checkpoint += 1

            stopping = early_stopping(val_loss=val_loss, epoch=self.epoch)
            if stopping:
                print('Early stopping. End of training.', flush=True)
                break

            self.lr_scheduler.step()
            self.epoch += 1

        # Construct the results dictionary
        result_dict = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'fid_scores': fid_scores,
            'mmds': mmds,
            'sigma': self.model.sigma.tolist(),
            'num_trained_batches': curr_num_batches,
            'trained_epochs': self.epoch,
        }
        torch.cuda.empty_cache()
        return result_dict


class FlowTester(object):
    def __init__(self,
                 model: FlowModel,
                 data: torch.Tensor,
                 *,
                 batch_size: int = 512,
                 criteria: Union[list, Tuple] = None,
                 complex_data: bool = True,
                 return_all_timesteps: bool = False,
                 fft_pre: bool = False,
                 mode: str = '1D'):
        """Tester class for a FlowModel instance."""
        self.model = model
        self.device = self.model.device
        self.complex_data = complex_data
        self.return_all_timesteps = return_all_timesteps
        self.fft_pre = fft_pre
        self.mode = mode

        # prepare test data
        self.num_samples, *data_shape = data.shape
        assert utils.equal_iterables(data_shape, self.model.data_shape)
        if self.fft_pre:
            # Transform for the network input
            data = ut.complex_1d_fft(data, ifft=False, mode=self.mode)
        self.data = data.to(dtype=torch.float32)
        self.dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=False, pin_memory=True)

        if self.fft_pre:
            # Transform back for the MSE evaluation
            self.data = ut.complex_1d_fft(data, ifft=True, mode=self.mode)

        # register all test functions for the requested criteria
        self.criteria = criteria
        self.test_funcs = [self._register_test_func(criterion) for criterion in criteria]

    def _register_test_func(self, criterion: str) -> callable:
        """Helper function to add specific test functions"""
        if criterion == 'nmse':
            return self._test_nmse
        elif criterion == 'fid':
            return self._test_fid
        else:
            raise NotImplementedError(criterion)

    def _test_fid(self):
        """Might be used to evaluate the FID score on the test data"""
        raise NotImplementedError

    def _test_mmd(self):
        """Might be used to evaluate the MMD on the test data"""
        raise NotImplementedError

    @torch.no_grad()
    def _test_nmse(self) -> dict:
        """Test function for the NMSE criterion."""
        # specify which SNRs should be evaluated: {0, 10, 20, 30} dB
        snr_db_range = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float32, device=self.device)
        snr_range = 10 ** (snr_db_range / 10)

        nmse_total_power_list = []
        timings_sec = []
        tps_ms_list = []

        with torch.no_grad():
            for snr in tqdm(iterable=snr_range):
                # COUNT TIME
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                # test each SNR value
                x_hat = []
                for data_batch in self.dataloader:
                    data_batch = data_batch.to(device=self.device)

                    # add noise to the test data
                    y = functional.awgn(data_batch, snr, multiplier=self.model.noise_multiplier)

                    # calculate channel estimate with num_steps=64 for Heun/RK2
                    x_est = self.model.generate_estimate(
                        y.to(device=self.device), snr, return_all_timesteps=self.return_all_timesteps, num_steps=64)

                    if self.fft_pre:
                        if self.return_all_timesteps:
                            x_est = ut.complex_1d_fft(x_est, ifft=True, mode=self.mode, _4d_array=True)
                        else:
                            x_est = ut.complex_1d_fft(x_est, ifft=True, mode=self.mode)

                    x_hat.append(x_est)

                x_hat = torch.cat(x_hat, dim=0).cpu()

                # COUNT TIME END
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                timings_sec.append(dt)
                tps_ms_list.append(dt * 1000.0 / self.num_samples)

                if self.return_all_timesteps:
                    nmse_total_power_list.append([])
                    n_timesteps = x_hat.shape[1]

                    if len(self.data.shape) == 5:
                        dim = int(self.data.shape[-1] * self.data.shape[-2])
                        x_hat_rs = ut.reshape_fortran(x_hat, (-1, n_timesteps, dim))
                        for t in range(n_timesteps):
                            nmse_total_power_list[-1].append(
                                functional.nmse_torch(
                                    ut.reshape_fortran(torch.squeeze(self.data), (-1, dim)),
                                    x_hat_rs[:, t],
                                    norm_per_sample=False
                                )
                            )
                    else:
                        for t in range(n_timesteps):
                            nmse_total_power_list[-1].append(
                                functional.nmse_torch(
                                    torch.squeeze(self.data),
                                    torch.squeeze(x_hat[:, t]),
                                    norm_per_sample=False
                                )
                            )
                else:
                    if len(self.data.shape) == 4:
                        dim = int(self.data.shape[-1] * self.data.shape[-2])
                        x_hat_rs = ut.reshape_fortran(x_hat, (-1, dim))
                        nmse_total_power_list.append(
                            functional.nmse_torch(
                                ut.reshape_fortran(torch.squeeze(self.data), (-1, dim)),
                                x_hat_rs,
                                norm_per_sample=False
                            )
                        )
                    else:
                        nmse_total_power_list.append(
                            functional.nmse_torch(
                                torch.squeeze(self.data),
                                torch.squeeze(x_hat),
                                norm_per_sample=False
                            )
                        )

        return {
            'SNRs': snr_db_range.tolist(),
            'NMSEs_total_power': nmse_total_power_list,
            'Timings_sec': timings_sec,
            'Time_per_sample_ms': tps_ms_list,
        }

    @torch.no_grad()
    def test(self) -> dict:
        """Main test function, intended for public use."""
        test_dict = {}
        self.model.eval()
        for criterion, test_func in zip(self.criteria, self.test_funcs):
            print(f'Testing criterion: "{criterion}"')
            test_dict[criterion] = test_func()
        return test_dict


class Early_stopping:
    def __init__(self, min_epochs: int = 1, num_epochs_no_improve: int = 1):
        self.min_epochs = min_epochs
        self.num_epochs_no_improve = num_epochs_no_improve
        self.best_val_loss = np.inf
        self.counter = num_epochs_no_improve

    def __call__(self, val_loss, epoch):
        if epoch > self.min_epochs:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = self.num_epochs_no_improve
            else:
                self.counter -= 1
                if self.counter < 1:
                    return True
        return False
