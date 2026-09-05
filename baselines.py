import modules.utils as ut
from modules.pilot_matrix import draw_xp_sqrt_gamma_identity_gaussian_numpy
import csv
import datetime
from estimators.lmmse import LMMSE, mp_eval
import numpy as np
import multiprocessing as mp
import os
import torch
import argparse


# --- MIMO pilot model: Y_P = H X_P + N (spatial pilots, vectorized y = A h + n) ---
#
# Pilot matrix: X_P = sqrt(γ) I_rect + sqrt(1-γ) G (γ=0: pure Gaussian; γ=1: rectangular I).
#
# Scov-LMMSE (global C from training vec(H)):
#   - Gaussian pilots: vec(Y)=A vec(H)+η with A = X_P^T ⊗ I_{n_rx}; n_y = n_rx * n_p.
#   - Orthonormal (identity): vec(y)=vec(H)+η with A = I; n_y = n_h = n_rx * n_tx.
#   Same σ² = 10^{-SNR/10} and same sample covariance C = E[vec(H)vec(H)^H] on training data.
#
# Genie-LMMSE: same A and σ² as above, but C = true prior covariance of vec(H) per sample
# (from Toeplitz / block structure in _genie_covariance_chunk). Gaussian path only had genie
# until identity genie was added below.

_MIMO_DEFAULT_GPU_CHUNK = 256
MIMO_DEFAULT_N_PILOTS = 16
# cuBLAS/MAGMA batched GETRF/TRSM warns and can be suboptimal when each system is large
# (e.g. n_y = n_rx * n_pilot with n_pilot=32). Use per-sample solves above this n_y.
_MIMO_MAX_NY_BATCHED_CUDA_SOLVE = 256


def _mimo_use_torch(device, *arrays):
    if device is not None:
        return True
    return any(isinstance(x, torch.Tensor) for x in arrays if x is not None)


def _mimo_torch_device(device, *fallback_tensors):
    if device is not None:
        return torch.device(device)
    for t in fallback_tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def _mimo_torch_cdtype(*refs):
    for r in refs:
        if isinstance(r, torch.Tensor):
            return r.dtype
        if r is not None:
            a = np.asarray(r)
            if a.dtype == np.complex128:
                return torch.complex128
    return torch.complex64


def _mimo_as_tensor(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _mimo_diag_epsilon(epsilon, use_torch):
    if epsilon is not None:
        return float(epsilon)
    return 1e-6 if use_torch else 0.0


def mimo_vec_channel(H):
    """Column-major vec(H), consistent with vec(H X_P) = (X_P^T ⊗ I) vec(H)."""
    H = np.asarray(H)
    return H.reshape(-1, order="F")


def mimo_unvec_channel(h, n_rx, n_tx):
    h = np.asarray(h)
    return h.reshape((n_rx, n_tx), order="F")


def mimo_vec_channel_torch(H):
    """Column-major vec: H (..., n_rx, n_tx) -> (..., n_rx * n_tx)."""
    return H.transpose(-1, -2).reshape(*H.shape[:-2], H.shape[-2] * H.shape[-1])


def mimo_unvec_channel_torch(h, n_rx, n_tx):
    """Unvec to (..., n_rx, n_tx), consistent with mimo_vec_channel_torch."""
    return h.reshape(*h.shape[:-1], n_tx, n_rx).transpose(-1, -2)


def mimo_pilot_linear_operator(X_P, n_rx):
    """
    A = X_P^T ⊗ I_{n_rx} so that vec(Y_P) = A vec(H) for Y_P = H X_P.
    X_P: (n_tx, n_p) -> A: (n_rx * n_p, n_rx * n_tx)
    """
    X_P = np.asarray(X_P)
    n_p = X_P.shape[1]
    eye = np.eye(n_rx, dtype=X_P.dtype)
    return np.kron(X_P.T, eye)


def mimo_pilot_linear_operator_torch(X_P, n_rx):
    """A = X_P^T ⊗ I_{n_rx} (torch). X_P: (n_tx, n_p)."""
    eye = torch.eye(n_rx, dtype=X_P.dtype, device=X_P.device)
    # transpose is non-contiguous; kron may use view internally
    x_t = X_P.transpose(-1, -2).contiguous()
    return torch.kron(x_t, eye)


def mimo_ls_estimate_torch(Y_P, X_P, epsilon=0.0):
    """
    Batched LS on GPU/CPU. Y_P: (B, n_rx, n_p) or (n_rx, n_p); X_P: (n_tx, n_p).
    H_hat = Y_P X_P^H (X_P X_P^H + epsilon I)^{-1}.
    """
    single = Y_P.dim() == 2
    if single:
        Y_P = Y_P.unsqueeze(0)
    nt = X_P.shape[0]
    eye = torch.eye(nt, dtype=X_P.dtype, device=X_P.device)
    M = X_P @ X_P.conj().transpose(-1, -2) + epsilon * eye
    R = Y_P @ X_P.conj().transpose(-1, -2)
    RH = R.conj().transpose(-1, -2)
    B = Y_P.shape[0]
    HH = torch.linalg.solve(M.unsqueeze(0).expand(B, -1, -1), RH)
    H = HH.conj().transpose(-1, -2)
    return H.squeeze(0) if single else H


def mimo_ls_estimate(Y_P, X_P, device=None, epsilon=0.0):
    """
    H_hat = Y_P X_P^H (X_P X_P^H)^{-1}. Y_P: (n_rx, n_p) or (B, n_rx, n_p).
    Pass device='cuda' (or a torch.device) for a batched GPU path; numpy if device is None and inputs are arrays.
    """
    if not _mimo_use_torch(device, Y_P, X_P):
        Y_P = np.asarray(Y_P)
        X_P = np.asarray(X_P)
        M = X_P @ X_P.conj().T + epsilon * np.eye(X_P.shape[0], dtype=X_P.dtype)
        R = Y_P @ X_P.conj().T
        if R.ndim == 2:
            return np.linalg.solve(M, R.conj().T).conj().T
        RH = np.swapaxes(R.conj(), -1, -2)
        HH = np.linalg.solve(M, RH)
        return np.swapaxes(HH.conj(), -1, -2)
    want_numpy = not isinstance(Y_P, torch.Tensor)
    dev = _mimo_torch_device(device, Y_P, X_P)
    dtype = _mimo_torch_cdtype(Y_P, X_P)
    Yt = _mimo_as_tensor(Y_P, dev, dtype)
    Xt = _mimo_as_tensor(X_P, dev, dtype)
    H = mimo_ls_estimate_torch(Yt, Xt, epsilon=epsilon)
    return H.cpu().numpy() if want_numpy else H


def sample_channel_covariance_from_vec(h_vecs):
    """
    C = (1/M) sum_m vec(h_m) vec(h_m)^H.
    h_vecs: (M, n_rx * n_tx)
    """
    h_vecs = np.asarray(h_vecs)
    M = h_vecs.shape[0]
    return (h_vecs.conj().T @ h_vecs) / M


def sample_channel_covariance_torch(h_vecs):
    """Same as sample_channel_covariance_from_vec; h_vecs: (M, D) tensor."""
    m = h_vecs.shape[0]
    return (h_vecs.conj().transpose(-1, -2) @ h_vecs) / m


def mimo_lmmse_estimate_batch_torch(y, A, C, sigma2, epsilon=0.0):
    """
    h_hat = C A^H (A C A^H + sigma^2 I + epsilon I)^{-1} y.
    y: (B, n_y), A: (n_y, n_h), C: (n_h, n_h).
    """
    ny = A.shape[0]
    eye = torch.eye(ny, dtype=A.dtype, device=A.device)
    S = A @ C @ A.conj().transpose(-1, -2) + sigma2 * eye + epsilon * eye
    CAh = C @ A.conj().transpose(-1, -2)
    sol = torch.linalg.solve(S, y.transpose(-1, -2))
    return (CAh @ sol).transpose(-1, -2)


def mimo_lmmse_genie_estimate_batch_torch(y, A, C_batch, sigma2, epsilon=0.0):
    """
    Per-sample C. y: (B, n_y), A: (n_y, n_h), C_batch: (B, n_h, n_h).
    """
    B, ny = y.shape
    nh = A.shape[1]
    eye_y = torch.eye(ny, dtype=A.dtype, device=A.device)
    A_exp = A.unsqueeze(0).expand(B, -1, -1)
    A_H = A.conj().transpose(-1, -2).unsqueeze(0).expand(B, -1, -1)
    AC = torch.bmm(A_exp, C_batch)
    S = torch.bmm(AC, A_H) + sigma2 * eye_y + epsilon * eye_y
    A_H_fixed = A.conj().transpose(-1, -2).unsqueeze(0).expand(B, -1, -1)
    CAh = torch.bmm(C_batch, A_H_fixed)
    # Batched torch.linalg.solve on CUDA uses MAGMA batched kernels tuned for small matrices.
    # Large n_y (e.g. 64*32) floods stderr with warnings; sequential solve uses classical path.
    if S.is_cuda and ny > _MIMO_MAX_NY_BATCHED_CUDA_SOLVE:
        sol = torch.stack(
            [torch.linalg.solve(S[b], y[b].unsqueeze(-1)) for b in range(B)],
            dim=0,
        )
    else:
        sol = torch.linalg.solve(S, y.unsqueeze(-1))
    return torch.bmm(CAh, sol).squeeze(-1)


def mimo_lmmse_estimate_from_y(y, A, C, sigma2, device=None, epsilon=None):
    """
    Single-sample LMMSE. y: (n_y,), A, C as in notes.
    """
    eps = _mimo_diag_epsilon(epsilon, _mimo_use_torch(device, y, A, C))
    if not _mimo_use_torch(device, y, A, C):
        y = np.asarray(y).ravel()
        A = np.asarray(A)
        C = np.asarray(C)
        S = A @ C @ A.conj().T + sigma2 * np.eye(A.shape[0], dtype=A.dtype) + eps * np.eye(A.shape[0], dtype=A.dtype)
        CAh = C @ A.conj().T
        return CAh @ np.linalg.solve(S, y)
    want_numpy = not isinstance(y, torch.Tensor)
    dev = _mimo_torch_device(device, y, A, C)
    dtype = _mimo_torch_cdtype(y, A, C)
    yt = _mimo_as_tensor(y, dev, dtype).reshape(1, -1)
    At = _mimo_as_tensor(A, dev, dtype)
    Ct = _mimo_as_tensor(C, dev, dtype)
    ht = mimo_lmmse_estimate_batch_torch(yt, At, Ct, sigma2, epsilon=eps)
    out = ht.squeeze(0)
    return out.cpu().numpy() if want_numpy else out


def mimo_lmmse_estimate_batch(y, A, C, sigma2, device=None, epsilon=None):
    """
    Batch LMMSE with shared A and C. y: (batch, n_y), returns h_hat: (batch, n_h).
    device='cuda' runs fully batched on GPU; epsilon defaults to 1e-6 on torch for stability.
    """
    eps = _mimo_diag_epsilon(epsilon, _mimo_use_torch(device, y, A, C))
    if not _mimo_use_torch(device, y, A, C):
        y = np.asarray(y)
        A = np.asarray(A)
        C = np.asarray(C)
        S = A @ C @ A.conj().T + sigma2 * np.eye(A.shape[0], dtype=A.dtype) + eps * np.eye(A.shape[0], dtype=A.dtype)
        CAh = C @ A.conj().T
        sol = np.linalg.solve(S, y.T)
        return (CAh @ sol).T
    want_numpy = not isinstance(y, torch.Tensor)
    dev = _mimo_torch_device(device, y, A, C)
    dtype = _mimo_torch_cdtype(y, A, C)
    yt = _mimo_as_tensor(y, dev, dtype)
    At = _mimo_as_tensor(A, dev, dtype)
    Ct = _mimo_as_tensor(C, dev, dtype)
    ht = mimo_lmmse_estimate_batch_torch(yt, At, Ct, sigma2, epsilon=eps)
    return ht.cpu().numpy() if want_numpy else ht


def mimo_lmmse_genie_estimate_batch(y, A, C_batch, sigma2, device=None, gpu_batch_size=None, epsilon=None):
    """
    Per-sample covariance C_batch[b]. y: (batch, n_y), C_batch: (batch, n_h, n_h) or (n_h, n_h).
    On GPU, uses batched solve; optional gpu_batch_size chunks large batches.
    """
    eps = _mimo_diag_epsilon(epsilon, _mimo_use_torch(device, y, A, C_batch))
    if not _mimo_use_torch(device, y, A, C_batch):
        y = np.asarray(y)
        A = np.asarray(A)
        C_batch = np.asarray(C_batch)
        single_C = C_batch.ndim == 2
        batch = y.shape[0]
        if single_C:
            C_batch = np.broadcast_to(C_batch, (batch,) + C_batch.shape)
        n_h = A.shape[1]
        h_hat = np.zeros((batch, n_h), dtype=np.result_type(y.dtype, A.dtype, C_batch.dtype))
        eye = eps * np.eye(A.shape[0], dtype=A.dtype)
        for b in range(batch):
            S = A @ C_batch[b] @ A.conj().T + sigma2 * np.eye(A.shape[0], dtype=A.dtype) + eye
            CAh = C_batch[b] @ A.conj().T
            h_hat[b] = CAh @ np.linalg.solve(S, y[b])
        return h_hat
    want_numpy = not isinstance(y, torch.Tensor)
    dev = _mimo_torch_device(device, y, A, C_batch)
    dtype = _mimo_torch_cdtype(y, A, C_batch)
    yt = _mimo_as_tensor(y, dev, dtype)
    At = _mimo_as_tensor(A, dev, dtype)
    Ct = _mimo_as_tensor(C_batch, dev, dtype)
    if Ct.dim() == 2:
        Ct = Ct.unsqueeze(0).expand(yt.shape[0], -1, -1)
    chunk = gpu_batch_size or _MIMO_DEFAULT_GPU_CHUNK
    outs = []
    for s in range(0, yt.shape[0], chunk):
        e = min(s + chunk, yt.shape[0])
        outs.append(
            mimo_lmmse_genie_estimate_batch_torch(yt[s:e], At, Ct[s:e], sigma2, epsilon=eps)
        )
    ht = torch.cat(outs, dim=0)
    return ht.cpu().numpy() if want_numpy else ht


def mimo_scov_lmmse_estimate_channels(Y_P, X_P, C_vec, sigma2, n_rx, n_tx, device=None, gpu_batch_size=None, epsilon=None):
    """
    Scov-LMMSE: global sample covariance C_vec on vec(H), same pilots for all rows of Y_P.
    Y_P: (batch, n_rx, n_p) or (n_rx, n_p). Use device='cuda' for GPU (chunked by gpu_batch_size).
    """
    use_torch = _mimo_use_torch(device, Y_P, X_P, C_vec)
    eps = _mimo_diag_epsilon(epsilon, use_torch)
    want_numpy = not isinstance(Y_P, torch.Tensor)
    if use_torch:
        dev = _mimo_torch_device(device, Y_P, X_P, C_vec)
        dtype = _mimo_torch_cdtype(Y_P, X_P, C_vec)
        Xt = _mimo_as_tensor(X_P, dev, dtype)
        Ct = _mimo_as_tensor(C_vec, dev, dtype)
        At = mimo_pilot_linear_operator_torch(Xt, n_rx)
        Yt = _mimo_as_tensor(Y_P, dev, dtype)
        single = Yt.dim() == 2
        if single:
            Yt = Yt.unsqueeze(0)
        if Yt.shape[-2] != n_rx:
            raise ValueError(f"n_rx mismatch: Y_P has {Yt.shape[-2]}, expected {n_rx}")
        batch = Yt.shape[0]
        chunk = gpu_batch_size or _MIMO_DEFAULT_GPU_CHUNK
        h_parts = []
        for s in range(0, batch, chunk):
            e = min(s + chunk, batch)
            y_chunk = mimo_vec_channel_torch(Yt[s:e])
            h_parts.append(mimo_lmmse_estimate_batch_torch(y_chunk, At, Ct, sigma2, epsilon=eps))
        h_hat = torch.cat(h_parts, dim=0)
        H_out = mimo_unvec_channel_torch(h_hat, n_rx, n_tx)
        if single:
            H_out = H_out.squeeze(0)
        return H_out.cpu().numpy() if want_numpy else H_out

    Y_P = np.asarray(Y_P)
    single = Y_P.ndim == 2
    if single:
        Y_P = Y_P[np.newaxis, ...]
    batch, n_rx_b, n_p = Y_P.shape
    if n_rx_b != n_rx:
        raise ValueError(f"n_rx mismatch: Y_P has {n_rx_b}, expected {n_rx}")
    A = mimo_pilot_linear_operator(X_P, n_rx)
    # Column-major vec(Y) to match A @ vec(H) and the torch path (mimo_vec_channel_torch).
    y = np.stack([mimo_vec_channel(Y_P[b]) for b in range(batch)], axis=0)
    h_hat = mimo_lmmse_estimate_batch(y, A, C_vec, sigma2, epsilon=eps)
    return mimo_unvec_channel(h_hat[0], n_rx, n_tx) if single else np.stack(
        [mimo_unvec_channel(h_hat[b], n_rx, n_tx) for b in range(batch)], axis=0
    )


def mimo_genie_lmmse_estimate_channels(
    Y_P, X_P, C_delta_batch, sigma2, n_rx, n_tx, device=None, gpu_batch_size=None, epsilon=None
):
    """
    Genie LMMSE: true C_delta per sample. C_delta_batch: (batch, n_rx*n_tx, n_rx*n_tx) or (n_h, n_h).
    """
    use_torch = _mimo_use_torch(device, Y_P, X_P, C_delta_batch)
    eps = _mimo_diag_epsilon(epsilon, use_torch)
    want_numpy = not isinstance(Y_P, torch.Tensor)
    if use_torch:
        dev = _mimo_torch_device(device, Y_P, X_P, C_delta_batch)
        dtype = _mimo_torch_cdtype(Y_P, X_P, C_delta_batch)
        Xt = _mimo_as_tensor(X_P, dev, dtype)
        Cdt = _mimo_as_tensor(C_delta_batch, dev, dtype)
        At = mimo_pilot_linear_operator_torch(Xt, n_rx)
        Yt = _mimo_as_tensor(Y_P, dev, dtype)
        single_Y = Yt.dim() == 2
        if single_Y:
            Yt = Yt.unsqueeze(0)
        if Yt.shape[-2] != n_rx:
            raise ValueError(f"n_rx mismatch: Y_P has {Yt.shape[-2]}, expected {n_rx}")
        batch = Yt.shape[0]
        if Cdt.dim() == 2:
            Cdt = Cdt.unsqueeze(0).expand(batch, -1, -1)
        y = mimo_vec_channel_torch(Yt)
        chunk = gpu_batch_size or _MIMO_DEFAULT_GPU_CHUNK
        h_parts = []
        for s in range(0, batch, chunk):
            e = min(s + chunk, batch)
            h_parts.append(
                mimo_lmmse_genie_estimate_batch_torch(y[s:e], At, Cdt[s:e], sigma2, epsilon=eps)
            )
        h_hat = torch.cat(h_parts, dim=0)
        H_out = mimo_unvec_channel_torch(h_hat, n_rx, n_tx)
        if single_Y:
            H_out = H_out.squeeze(0)
        return H_out.cpu().numpy() if want_numpy else H_out

    Y_P = np.asarray(Y_P)
    C_delta_batch = np.asarray(C_delta_batch)
    single_C = C_delta_batch.ndim == 2
    single_Y = Y_P.ndim == 2
    if single_Y:
        Y_P = Y_P[np.newaxis, ...]
    batch, n_rx_b, n_p = Y_P.shape
    if n_rx_b != n_rx:
        raise ValueError(f"n_rx mismatch: Y_P has {n_rx_b}, expected {n_rx}")
    A = mimo_pilot_linear_operator(X_P, n_rx)
    y = np.stack([mimo_vec_channel(Y_P[b]) for b in range(batch)], axis=0)
    if single_C:
        C_batch = np.broadcast_to(C_delta_batch, (batch,) + C_delta_batch.shape)
    else:
        C_batch = C_delta_batch
    h_hat = mimo_lmmse_genie_estimate_batch(y, A, C_batch, sigma2, device=device, gpu_batch_size=gpu_batch_size, epsilon=eps)
    return mimo_unvec_channel(h_hat[0], n_rx, n_tx) if single_Y else np.stack(
        [mimo_unvec_channel(h_hat[b], n_rx, n_tx) for b in range(batch)], axis=0
    )


def mimo_pilot_observation(H, X_P, snr_db, rng=None, device=None, generator=None):
    """
    Y_P = H X_P + N with N ~ CN(0, sigma^2 I), matched to ut.get_observation scaling
    (per-entry noise std 10^{-SNR/20}). Pass device='cuda' for torch on GPU.
    """
    sigma = 10 ** (-snr_db / 20.0)
    scale = sigma * np.sqrt(0.5)
    if not _mimo_use_torch(device, H, X_P):
        rng = np.random.default_rng(rng)
        H = np.asarray(H)
        X_P = np.asarray(X_P)
        shape = (H @ X_P).shape
        Y = H @ X_P + scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        return Y
    want_numpy = not isinstance(H, torch.Tensor)
    dev = _mimo_torch_device(device, H, X_P)
    dtype = _mimo_torch_cdtype(H, X_P)
    Ht = _mimo_as_tensor(H, dev, dtype)
    Xt = _mimo_as_tensor(X_P, dev, dtype)
    Y0 = Ht @ Xt
    g = generator
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    nr = torch.randn(Y0.shape, device=dev, dtype=real_dtype, generator=g)
    ni = torch.randn(Y0.shape, device=dev, dtype=real_dtype, generator=g)
    noise = torch.complex(nr, ni).to(dtype=dtype) * scale
    Y = Y0 + noise
    return Y.cpu().numpy() if want_numpy else Y


def mp_gmm(obj, *args):
    return obj.estimate_from_y(*args)

def mp_omp(obj, *args):
    return obj.estimate(*args)


def _genie_covariance_chunk(toep_test, b0, b1, n_rx, n_tx, ch_type):
    """
    Oracle covariances C_delta for vec(H), shape (chunk, n_rx*n_tx, n_rx*n_tx).
    Matches LMMSE genie structure: Kronecker (3gpp-style tuple), block-diagonal
    (pseudo_multiuser), or single Toeplitz.
    """
    nh = n_rx * n_tx
    chunk = b1 - b0
    dtype = np.complex128
    if isinstance(toep_test, tuple):
        t_rx, t_tx = toep_test
        if ch_type.startswith("pseudo_multiuser") and t_rx.ndim == 3:
            C_chunk = np.zeros((chunk, nh, nh), dtype=dtype)
            for bi, b in enumerate(range(b0, b1)):
                off = 0
                for k in range(n_tx):
                    C_rx = ut.toeplitz(t_rx[b, k, :])
                    nk = C_rx.shape[0]
                    C_chunk[bi, off : off + nk, off : off + nk] = C_rx
                    off += nk
                if off != nh:
                    raise ValueError(f"genie cov size mismatch: off={off}, nh={nh}")
            return C_chunk
        C_chunk = np.zeros((chunk, nh, nh), dtype=dtype)
        for bi, b in enumerate(range(b0, b1)):
            C_rx = ut.toeplitz(np.asarray(t_rx[b]).ravel())
            C_tx = ut.toeplitz(np.asarray(t_tx[b]).ravel())
            C_chunk[bi] = np.kron(C_tx, C_rx)
        return C_chunk
    t = toep_test
    C_chunk = np.zeros((chunk, nh, nh), dtype=dtype)
    for bi, b in enumerate(range(b0, b1)):
        C_chunk[bi] = ut.toeplitz(t[b, :]).T
    return C_chunk


def _nmse_accumulate(H_true, H_hat):
    """Same normalization as legacy baselines: ||H-Ĥ||_F^2 / ||H||_F^2 on flattened channels."""
    diff = np.asarray(H_true) - np.asarray(H_hat)
    return np.sum(np.abs(diff) ** 2), np.sum(np.abs(H_true) ** 2)


def run_mimo_pilot_baselines_single_snr(
    channels_train_flat,
    channels_test_flat,
    toep_test,
    snr_db,
    n_rx,
    n_tx,
    ch_type,
    n_pilots=None,
    device="cuda",
    gpu_batch_size=64,
    pilot_seed=0,
    noise_seed_base=0,
    epsilon=None,
    run_ls=True,
    run_scov=True,
    run_genie=True,
    spatial_pilot_gamma=0.0,
    pilot_power_norm="legacy",
):
    """
    MIMO spatial-pilot baselines at one SNR (X_P = sqrt(γ) I_rect + sqrt(1-γ) G).
    Enable subsets with run_ls / run_scov / run_genie.
    Returns dict with NMSE only for methods that ran, plus snr_db, n_pilots, n_test.
    """
    n_pilots = n_pilots if n_pilots is not None else MIMO_DEFAULT_N_PILOTS
    nh = n_rx * n_tx
    rng_xp = np.random.default_rng(pilot_seed)
    X_P = draw_xp_sqrt_gamma_identity_gaussian_numpy(
        n_tx,
        n_pilots,
        float(spatial_pilot_gamma),
        rng_xp,
        power_norm=str(pilot_power_norm),
    )

    C_scov = None
    if run_scov:
        cht = np.asarray(channels_train_flat)
        if cht.ndim == 2 and cht.shape[1] == nh:
            train_vecs = cht
        else:
            train_vecs = np.stack(
                [mimo_vec_channel(np.reshape(cht[i], (n_rx, n_tx), order="F")) for i in range(cht.shape[0])]
            )
        C_scov = sample_channel_covariance_from_vec(train_vecs)

    sigma2 = 10 ** (-snr_db / 10.0)
    n_test = channels_test_flat.shape[0]
    _noise_seed = int(noise_seed_base) + int(snr_db) * 1009 + 17
    rng_noise = np.random.default_rng(_noise_seed % (2**63 - 1))

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev_arg = "cuda" if use_cuda else None
    chunk = min(gpu_batch_size, n_test)

    sum_ls = sum_scov = sum_genie = 0.0
    den = 0.0
    genie_cap = min(chunk, 48, max(8, gpu_batch_size))

    for s in range(0, n_test, chunk):
        e = min(s + chunk, n_test)
        H = np.reshape(channels_test_flat[s:e], (-1, n_rx, n_tx), order="F")
        den += np.sum(np.abs(H) ** 2)
        Y_P = mimo_pilot_observation(H, X_P, snr_db, rng=rng_noise)
        if run_ls:
            H_ls = mimo_ls_estimate(Y_P, X_P, device=dev_arg, epsilon=0.0)
            sum_ls += _nmse_accumulate(H, H_ls)[0]
        if run_scov:
            H_scov = mimo_scov_lmmse_estimate_channels(
                Y_P, X_P, C_scov, sigma2, n_rx, n_tx, device=dev_arg, gpu_batch_size=chunk, epsilon=epsilon
            )
            sum_scov += _nmse_accumulate(H, H_scov)[0]
        if run_genie:
            C_gen = _genie_covariance_chunk(toep_test, s, e, n_rx, n_tx, ch_type)
            H_gen = mimo_genie_lmmse_estimate_channels(
                Y_P, X_P, C_gen, sigma2, n_rx, n_tx, device=dev_arg, gpu_batch_size=genie_cap, epsilon=epsilon
            )
            sum_genie += _nmse_accumulate(H, H_gen)[0]

    out = {
        "snr_db": float(snr_db),
        "n_pilots": int(n_pilots),
        "n_test": int(n_test),
        "spatial_pilot_gamma": float(spatial_pilot_gamma),
    }
    if run_ls:
        out["mimo_ls"] = float(sum_ls / den)
    if run_scov:
        out["mimo_scov_lmmse"] = float(sum_scov / den)
    if run_genie:
        out["mimo_genie_lmmse"] = float(sum_genie / den)
    return out


def _identity_awgn_observation(H, snr_db, rng):
    """
    y = H + n, same distribution as ut.get_observation(H, snr_db, A=None).
    Uses explicit RNG for reproducible sweeps (matches noise model in utils.get_observation).
    """
    sigma = 10 ** (-snr_db / 20.0)
    H = np.asarray(H)
    noise = sigma * ut.crandn(*H.shape, rng=rng)
    return H + noise


def run_identity_lmmse_baselines_single_snr(
    channels_train_flat,
    channels_test_flat,
    toep_test,
    snr_db,
    n_rx,
    n_tx,
    ch_type,
    device="cuda",
    gpu_batch_size=64,
    noise_seed_base=0,
    epsilon=None,
    run_scov=True,
    run_genie=False,
):
    """
    Orthonormal / identity pilot: vec(y) = vec(H) + η (y = H + n element-wise, same noise as ut.get_observation).

    Scov: ĥ = C_scov (C_scov + σ² I)^{-1} y with global C_scov from training vec(H) (same construction as Gaussian Scov).

    Genie: ĥ = C_b (C_b + σ² I)^{-1} y with true C_b = Cov(vec(H)|prior) from _genie_covariance_chunk (same as Gaussian genie, but A = I).
    """
    if not run_scov and not run_genie:
        raise ValueError("run_identity_lmmse_baselines_single_snr: set at least one of run_scov, run_genie")
    if run_genie and toep_test is None:
        raise ValueError("run_genie=True requires toep_test (genie covariance)")

    nh = n_rx * n_tx
    C_scov = None
    if run_scov:
        cht = np.asarray(channels_train_flat)
        if cht.ndim == 2 and cht.shape[1] == nh:
            train_vecs = cht
        else:
            train_vecs = np.stack(
                [mimo_vec_channel(np.reshape(cht[i], (n_rx, n_tx), order="F")) for i in range(cht.shape[0])]
            )
        C_scov = sample_channel_covariance_from_vec(train_vecs)

    sigma2 = 10 ** (-snr_db / 10.0)
    A = np.eye(nh, dtype=np.complex128)

    n_test = channels_test_flat.shape[0]
    _noise_seed = int(noise_seed_base) + int(snr_db) * 1009 + 17
    rng_noise = np.random.default_rng(_noise_seed % (2**63 - 1))

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev_arg = "cuda" if use_cuda else None
    chunk = min(gpu_batch_size, n_test)
    genie_cap = min(chunk, 48, max(8, gpu_batch_size))

    sum_scov = sum_genie = 0.0
    den = 0.0

    for s in range(0, n_test, chunk):
        e = min(s + chunk, n_test)
        H = np.reshape(channels_test_flat[s:e], (-1, n_rx, n_tx), order="F")
        den += np.sum(np.abs(H) ** 2)
        Y_obs = _identity_awgn_observation(H, snr_db, rng_noise)
        batch = Y_obs.shape[0]
        y_vec = np.stack([mimo_vec_channel(Y_obs[b]) for b in range(batch)], axis=0)
        if run_scov:
            h_hat = mimo_lmmse_estimate_batch(y_vec, A, C_scov, sigma2, device=dev_arg, epsilon=epsilon)
            H_hat = np.stack([mimo_unvec_channel(h_hat[b], n_rx, n_tx) for b in range(batch)], axis=0)
            sum_scov += _nmse_accumulate(H, H_hat)[0]
        if run_genie:
            C_gen = _genie_covariance_chunk(toep_test, s, e, n_rx, n_tx, ch_type)
            h_gen = mimo_lmmse_genie_estimate_batch(
                y_vec, A, C_gen, sigma2, device=dev_arg, gpu_batch_size=genie_cap, epsilon=epsilon
            )
            H_gen = np.stack([mimo_unvec_channel(h_gen[b], n_rx, n_tx) for b in range(batch)], axis=0)
            sum_genie += _nmse_accumulate(H, H_gen)[0]

    out = {"snr_db": float(snr_db), "n_test": int(n_test)}
    if run_scov:
        out["identity_scov_lmmse"] = float(sum_scov / den)
    if run_genie:
        out["identity_genie_lmmse"] = float(sum_genie / den)
    return out


def _snr_grid_db(snr_min: float, snr_max: float, snr_step: float) -> list:
    """Inclusive SNR grid in dB (same arange convention as run_gram_oracle_3gpp_nmse.py)."""
    if snr_step <= 0:
        raise ValueError("--snr_step must be positive")
    if snr_min > snr_max + 1e-12:
        raise ValueError("--snr_min must be <= --snr_max")
    grid = np.arange(float(snr_min), float(snr_max) + 1e-9, float(snr_step), dtype=float).tolist()
    if not grid:
        raise ValueError("SNR grid is empty; check --snr_min, --snr_max, --snr_step")
    return grid


def main():
    parser = argparse.ArgumentParser(description='Run baseline channel estimation methods')
    parser.add_argument('--device', '-d', default='cuda', type=str, 
                       help='Device to use: cpu or cuda (default: cpu)')
    parser.add_argument('--gpu_batch_size', type=int, default=None,
                       help='Batch size for GPU processing (default: auto-detect based on GPU memory)')
    parser.add_argument('--snr_idx', type=int, default=None,
                       help='If set, only run this index into the SNR grid from --snr_min/--snr_max/--snr_step (for SLURM arrays).')
    parser.add_argument(
        '--snr_db',
        type=float,
        default=None,
        help='MIMO / identity baselines: run a single SNR in dB (e.g. 30). Overrides --snr_idx and the SNR grid.',
    )
    parser.add_argument(
        '--snr_min',
        type=float,
        default=-15.0,
        help='Start of SNR grid (dB); same pattern as run_gram_oracle_3gpp_nmse.py.',
    )
    parser.add_argument(
        '--snr_max',
        type=float,
        default=5.0,
        help='End of SNR grid (dB), inclusive.',
    )
    parser.add_argument(
        '--snr_step',
        type=float,
        default=1.0,
        help='SNR step (dB). Default 1 reproduces the previous [-15..5] sweep.',
    )
    parser.add_argument('--mimo_pilot', action='store_true',
                       help='Run spatial-pilot MIMO baselines (LS, Scov-LMMSE, Genie LMMSE); X_P from --spatial_pilot_gamma.')
    parser.add_argument(
        '--mimo_identity_scov',
        action='store_true',
        help='Orthonormal pilot: y = H + AWGN, Scov-LMMSE (A=I). Combine with --mimo_identity_genie for genie LMMSE.',
    )
    parser.add_argument(
        '--mimo_identity_genie',
        action='store_true',
        help='Orthonormal pilot: same y=H+n as Scov, but Genie LMMSE with true C(vec(H)) from Toeplitz (A=I). '
             'Can use with or without --mimo_identity_scov.',
    )
    parser.add_argument('--mimo_ls', action='store_true', help='MIMO pilot: run LS only (combine with --mimo_scov / --mimo_genie for subsets).')
    parser.add_argument('--mimo_scov', action='store_true', help='MIMO pilot: run sample-covariance LMMSE.')
    parser.add_argument('--mimo_genie', action='store_true', help='MIMO pilot: run genie LMMSE.')
    parser.add_argument('--mimo_save_out', action='store_true',
                       help='Write one combined summary .out per SNR under --output_dir (default: terminal only).')
    parser.add_argument('--output_dir', type=str, default='./results/baselines/mimo_pilot',
                       help='Directory for .out files when --mimo_save_out is set.')
    parser.add_argument('--pilot_seed', type=int, default=0, help='RNG seed for fixed pilot matrix X_P.')
    parser.add_argument('--noise_seed_base', type=int, default=0, help='Base seed for pilot noise (SNR folded in).')
    parser.add_argument(
        '--n_pilots',
        type=int,
        default=None,
        help=f'Number of pilot symbols n_p (default: {MIMO_DEFAULT_N_PILOTS}).',
    )
    parser.add_argument(
        '--spatial_pilot_gamma',
        type=float,
        default=0.0,
        help='MIMO pilot blend γ in [0,1]: X_P = sqrt(γ)*I_rect + sqrt(1-γ)*G (CN pilots), same as load_and_eval_dm_dps.',
    )
    parser.add_argument(
        '--pilot_power_norm',
        type=str,
        default='legacy',
        choices=['legacy', 'align_i', 'row_norm'],
        help='Power normalization for MIMO pilots. '
             '"legacy": keep historical scaling (backward compatible). '
             '"align_i": scale Gaussian part so that E[G G^H]=I when γ=0. '
             '"row_norm": same as align_i then L2-normalize each row of final X_P.',
    )
    parser.add_argument('--run_legacy', action='store_true',
                       help='After --mimo_pilot, also run the original identity-observation baselines in main().')
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda':
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
            print(f'GPU Memory: {gpu_memory_gb:.1f} GB')
            
            # Auto-detect batch size if not specified
            if args.gpu_batch_size is None:
                # Estimate: each channel needs ~50-60MB, leave 2GB for system/PyTorch overhead
                available_memory_gb = gpu_memory_gb - 2.0
                estimated_batch_size = int(available_memory_gb * 1024 / 60)  # 60MB per channel
                # Clamp to reasonable range
                args.gpu_batch_size = max(10, min(estimated_batch_size, 200))
                print(f'Auto-detected GPU batch_size: {args.gpu_batch_size} (based on {gpu_memory_gb:.1f}GB GPU)')
            else:
                print(f'Using specified GPU batch_size: {args.gpu_batch_size}')
        else:
            print('WARNING: CUDA requested but not available, falling back to CPU')
            device = 'cpu'
    
    if device == 'cpu':
        n_processes = int(mp.cpu_count() / 2)
        print(f'Using CPU with {n_processes} processes')
        # prepare multiprocessing
        pool = mp.Pool(processes=n_processes)
    else:
        pool = None  # Not needed for GPU

    if args.mimo_pilot or args.mimo_identity_scov or args.mimo_identity_genie:
        if args.mimo_pilot and not (0.0 <= float(args.spatial_pilot_gamma) <= 1.0):
            raise ValueError("--spatial_pilot_gamma must be in [0, 1].")
        snrs_grid = _snr_grid_db(args.snr_min, args.snr_max, args.snr_step)
        if args.snr_db is not None:
            if args.snr_idx is not None:
                raise ValueError("Use either --snr_db or --snr_idx, not both.")
            snr_list_mimo = [float(args.snr_db)]
        elif args.snr_idx is not None:
            if args.snr_idx < 0 or args.snr_idx >= len(snrs_grid):
                raise ValueError(f"--snr_idx must be in [0, {len(snrs_grid) - 1}], got {args.snr_idx}")
            snr_list_mimo = [snrs_grid[args.snr_idx]]
        else:
            snr_list_mimo = snrs_grid

        n_arx = 64
        n_atx = 16
        n_tr = 100_000
        n_val = 10_000
        n_te = 10_000
        ch_mimo = "pseudo_multiuser_3gpp"
        path_mimo = 3
        ch_train, _, _, _, ch_test, toep_te = ut.load_or_create_data(
            ch_type=ch_mimo,
            n_path=path_mimo,
            n_antennas_rx=n_arx,
            n_antennas_tx=n_atx,
            n_train_ch=n_tr,
            n_val_ch=n_val,
            n_test_ch=n_te,
            return_toep=True,
        )
        n_ant = n_arx * n_atx
        if ch_mimo.startswith("quadriga"):
            ch_train = np.reshape(ch_train, (-1, n_ant), "F")
            ch_test = np.reshape(ch_test, (-1, n_ant), "F")

        gbs_mimo = args.gpu_batch_size if args.gpu_batch_size is not None else 64
        jid = os.environ.get("SLURM_JOB_ID", "local")
        aid = os.environ.get("SLURM_ARRAY_TASK_ID", str(args.snr_idx if args.snr_idx is not None else "all"))

        if args.mimo_save_out:
            os.makedirs(args.output_dir, exist_ok=True)

        def _snr_tag(snr_v: float) -> str:
            s = f"{snr_v:g}"
            return s.replace(".", "p").replace("+", "p").replace("-", "m")

        if args.mimo_pilot:
            any_mimo_flag = args.mimo_ls or args.mimo_scov or args.mimo_genie
            run_ls = args.mimo_ls if any_mimo_flag else True
            run_scov = args.mimo_scov if any_mimo_flag else True
            run_genie = args.mimo_genie if any_mimo_flag else True

            key_label = [
                ("mimo_ls", "LS"),
                ("mimo_scov_lmmse", "ScovLMMSE"),
                ("mimo_genie_lmmse", "GenieLMMSE"),
            ]

            for snr_db in snr_list_mimo:
                metrics = run_mimo_pilot_baselines_single_snr(
                    ch_train,
                    ch_test,
                    toep_te,
                    snr_db,
                    n_arx,
                    n_atx,
                    ch_mimo,
                    n_pilots=args.n_pilots,
                    device=device,
                    gpu_batch_size=gbs_mimo,
                    pilot_seed=args.pilot_seed,
                    noise_seed_base=args.noise_seed_base,
                    epsilon=None,
                    run_ls=run_ls,
                    run_scov=run_scov,
                    run_genie=run_genie,
                    spatial_pilot_gamma=float(args.spatial_pilot_gamma),
                    pilot_power_norm=str(args.pilot_power_norm),
                )
                if args.snr_db is not None:
                    idx_out = "custom"
                elif args.snr_idx is not None:
                    idx_out = args.snr_idx
                else:
                    idx_out = snrs_grid.index(snr_db)
                line_parts = [f"SNR={snr_db} dB", f"idx={idx_out}"]
                for key, label in key_label:
                    if key in metrics:
                        line_parts.append(f"{label}={metrics[key]:.6e}")
                print("[MIMO pilot]", " | ".join(line_parts), flush=True)

                if args.mimo_save_out:
                    snr_tag = _snr_tag(float(snr_db))
                    summary_path = os.path.join(
                        args.output_dir,
                        f"mimo_pilot_job{jid}_a{aid}_snr{snr_tag}dB.out",
                    )
                    with open(summary_path, "w") as sf:
                        sf.write(f"SNR_dB={snr_db}\nSNR_idx={idx_out}\n")
                        for k, v in metrics.items():
                            sf.write(f"{k}={v}\n")

        if args.mimo_identity_scov or args.mimo_identity_genie:
            for snr_db in snr_list_mimo:
                metrics = run_identity_lmmse_baselines_single_snr(
                    ch_train,
                    ch_test,
                    toep_te,
                    snr_db,
                    n_arx,
                    n_atx,
                    ch_mimo,
                    device=device,
                    gpu_batch_size=gbs_mimo,
                    noise_seed_base=args.noise_seed_base,
                    epsilon=None,
                    run_scov=bool(args.mimo_identity_scov),
                    run_genie=bool(args.mimo_identity_genie),
                )
                if args.snr_db is not None:
                    idx_out = "custom"
                elif args.snr_idx is not None:
                    idx_out = args.snr_idx
                else:
                    idx_out = snrs_grid.index(snr_db)
                line_parts = [f"SNR={snr_db} dB", f"idx={idx_out}"]
                if "identity_scov_lmmse" in metrics:
                    line_parts.append(f"ScovLMMSE={metrics['identity_scov_lmmse']:.6e}")
                if "identity_genie_lmmse" in metrics:
                    line_parts.append(f"GenieLMMSE={metrics['identity_genie_lmmse']:.6e}")
                print("[MIMO identity y=H+n]", " | ".join(line_parts), flush=True)

                if args.mimo_save_out:
                    snr_tag = _snr_tag(float(snr_db))
                    summary_path = os.path.join(
                        args.output_dir,
                        f"mimo_identity_job{jid}_a{aid}_snr{snr_tag}dB.out",
                    )
                    with open(summary_path, "w") as sf:
                        sf.write(f"SNR_dB={snr_db}\nSNR_idx={idx_out}\n")
                        for k, v in metrics.items():
                            sf.write(f"{k}={v}\n")

        if not args.run_legacy:
            if pool is not None:
                pool.close()
                pool.join()
            return

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    n_antennas_rx = 64
    n_antennas_tx = 16
    n_train_ch = 100_000
    n_val_ch = 10_000  # must not exceed size of training set
    n_test_ch = 10_000
    snrs = _snr_grid_db(args.snr_min, args.snr_max, args.snr_step)
    # ch_type = 'quadriga_LOS'
    # ch_type = '3gpp'
    ch_type = 'pseudo_multiuser_3gpp'
    n_path = 3

    eval_LS = False
    eval_lmmse_glob = False
    eval_lmmse_genie = True

    channels_train, toep_train, channels_val, _, channels_test, toep_test = ut.load_or_create_data(ch_type=ch_type,
                            n_path=n_path, n_antennas_rx=n_antennas_rx, n_antennas_tx=n_antennas_tx,
                            n_train_ch=n_train_ch, n_val_ch=n_val_ch, n_test_ch=n_test_ch, return_toep=True)

    # vectorize channels
    n_antennas = n_antennas_rx * n_antennas_tx
    if ch_type.startswith('quadriga'):
        channels_train = np.reshape(channels_train, (-1, n_antennas), 'F')
        channels_test = np.reshape(channels_test, (-1, n_antennas), 'F')

    mse_list = list()
    mse_list.append(snrs.copy())
    mse_list[-1].insert(0, 'SNR')

    if eval_lmmse_glob:
        print('Computing lmmse_glob...')
        mse_list.append(['lmmse_glob'])
        print('  Computing covariance matrix...')
        cov = np.zeros([n_antennas, n_antennas], dtype=complex)
        for i in range(channels_train.shape[0]):
            cov = cov + np.expand_dims(channels_train[i, :], 1) @ np.expand_dims(channels_train[i, :].conj(), 0)
        cov = cov / channels_train.shape[0]
        print('  Evaluating lmmse_glob for all SNRs...')
        if device == 'cpu' and pool is not None:
            eval_list_glob = list()
            for snr in snrs:
                y = ut.get_observation(channels_test, snr)
                eval_list_glob.append([LMMSE(snr), y, cov, False])
            res_glob_lmmse = pool.starmap(mp_eval, eval_list_glob)
        else:
            # CPU sequential (or could also use GPU, but lmmse_glob is fast)
            res_glob_lmmse = []
            for snr in snrs:
                y = ut.get_observation(channels_test, snr)
                res = mp_eval(LMMSE(snr), y, cov, False)
                res_glob_lmmse.append(res)
        for it, res in enumerate(res_glob_lmmse):
            mse_act = np.sum(np.abs(res - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)
        print('  lmmse_glob done.')


    if eval_LS:
        print('Computing LS...')
        mse_list.append(['LS'])
        for snr in snrs:
            y = ut.get_observation(channels_test, snr)
            mse_act = np.sum(np.abs(y - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)
        print('  LS done.')


    if ch_type.endswith('3gpp') and eval_lmmse_genie:
        print('Computing lmmse_genie...')
        print(f'  Processing {len(snrs)} SNRs × {n_test_ch} test channels = {len(snrs) * n_test_ch} total operations')
        if device == 'cuda':
            print(f'  Using GPU with batch_size={args.gpu_batch_size}')
        else:
            print('  Using CPU multiprocessing')
        mse_list.append(['lmmse_genie'])
        res_genie_lmmse = []
        
        for snr_idx, snr in enumerate(snrs):
            print(f'  Processing SNR={snr}dB ({snr_idx+1}/{len(snrs)})...')
            y = ut.get_observation(channels_test, snr)
            
            if device == 'cuda':
                # Use GPU version with batch processing
                lmmse_obj = LMMSE(snr)
                if ch_type.startswith('pseudo_multiuser'):
                    res = lmmse_obj.estimate_genie_independent_cols_gpu(y, toep_test, device=device, 
                                                                        batch_size=args.gpu_batch_size)
                else:
                    res = lmmse_obj.estimate_genie_gpu(y, toep_test, device=device, 
                                                        batch_size=args.gpu_batch_size)
            else:
                # Use CPU multiprocessing
                is_independent = ch_type.startswith('pseudo_multiuser')
                if pool is not None:
                    res = pool.starmap(mp_eval, [[LMMSE(snr), y, toep_test, True, None, is_independent]])[0]
                else:
                    res = mp_eval(LMMSE(snr), y, toep_test, True, None, is_independent)
            
            res_genie_lmmse.append(res)
            mse_act = np.sum(np.abs(res - channels_test) ** 2) / np.sum(np.abs(channels_test) ** 2)
            mse_list[-1].append(mse_act)
            print(f'    SNR={snr}dB: MSE={mse_act:.6e}')
        
        print('  lmmse_genie done.')


    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    os.makedirs('./results/baselines/', exist_ok=True)
    file_name = f'./results/baselines/{date_time}_{ch_type}_path={n_path}_ant={n_antennas_rx}x{n_antennas_tx}_' \
                f'testdata={channels_test.shape[0]}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)
    
    # Clean up
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()

