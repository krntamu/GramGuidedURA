import numpy as np
import torch
from modules.utils import toeplitz


def mp_eval(obj, y, toep, genie, A=None):
    if genie:
        hest = obj.estimate_genie(y, toep, A)
    else:
        hest = obj.estimate_global(y, toep, A)
    return hest #return_mse(hest, h_true)


class LMMSE:
    def __init__(self, snr):
        self.snr = snr
        self.rho = 10 ** (0.1 * snr)
        self.sigma2 = 1 / self.rho

    def estimate_genie(self, y, t, A=None):
        (n_batches, n_antennas) = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=y.dtype)
        hest = np.zeros([y.shape[0], A.shape[1]], dtype=y.dtype)
        for b in range(n_batches):
            if type(t) is tuple:
                t_rx, t_tx = t
                C_rx = toeplitz(t_rx[b, :])
                C_tx = toeplitz(t_tx[b, :])
                C = np.kron(C_tx, C_rx)
            else:
                C = toeplitz(t[b, :]).T  # get full cov matrix
            CAh = C @ A.conj().T
            Cy = A @ CAh + 1 / self.rho * np.eye(A.shape[0])
            # Use solve instead of pinv for better performance (only need to solve linear system)
            hest[b, :] = CAh @ np.linalg.solve(Cy, y[b, :])
        return hest


    def estimate_global(self, y, C, A=None):
        (n_batches, n_antennas) = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=complex)
        hest = np.zeros([y.shape[0], A.shape[1]], dtype=y.dtype)
        Cy = A @ C @ A.conj().T + 1 / self.rho * np.eye(A.shape[0])
        Cinv = np.linalg.pinv(Cy, hermitian=True)
        prod = C @ A.conj().T @ Cinv
        for b in range(n_batches):
            hest[b, :] = prod @ y[b, :]
        return hest

    def estimate_genie_gpu(self, y, t, device='cuda', batch_size=100, A=None):
        """
        GPU-accelerated version of estimate_genie with batch processing.
        
        Args:
            y: numpy array of shape (n_batches, n_antennas) - observations
            t: Toeplitz parameters, either tuple (t_rx, t_tx) or array
            device: 'cuda' or 'cpu'
            batch_size: number of channels to process in parallel
            A: observation matrix (optional)
        
        Returns:
            numpy array of shape (n_batches, n_antennas) - channel estimates
        """
        import torch
        
        n_batches, n_antennas = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=y.dtype)
        
        # Determine dtype: NumPy complex128 -> torch.complex128, complex64 -> torch.complex64
        if y.dtype == np.complex128:
            torch_dtype = torch.complex128
        elif y.dtype == np.complex64:
            torch_dtype = torch.complex64
        else:
            # Default to complex128 for better precision
            torch_dtype = torch.complex128
        
        # Convert to torch tensors with consistent dtype
        y_torch = torch.from_numpy(y).to(dtype=torch_dtype, device=device)
        A_torch = torch.from_numpy(A).to(dtype=torch_dtype, device=device)
        rho_torch = torch.tensor(self.rho, dtype=torch_dtype, device=device)
        
        hest_list = []
        
        # Process in batches to manage GPU memory
        for batch_start in range(0, n_batches, batch_size):
            batch_end = min(batch_start + batch_size, n_batches)
            batch_indices = slice(batch_start, batch_end)
            batch_size_actual = batch_end - batch_start
            
            y_batch = y_torch[batch_indices]  # (batch_size, n_antennas)
            
            if type(t) is tuple:
                t_rx, t_tx = t
                t_rx_batch = torch.from_numpy(t_rx[batch_indices]).to(dtype=torch_dtype, device=device)  # (batch_size, n_rx)
                t_tx_batch = torch.from_numpy(t_tx[batch_indices]).to(dtype=torch_dtype, device=device)  # (batch_size, n_tx)
                
                # Build Toeplitz matrices for all channels in batch
                n_rx = t_rx_batch.shape[1]
                n_tx = t_tx_batch.shape[1]
                
                # Build C_rx: (batch_size, n_rx, n_rx)
                C_rx = self._toeplitz_batch(t_rx_batch)
                
                # Build C_tx: (batch_size, n_tx, n_tx)
                C_tx = self._toeplitz_batch(t_tx_batch)
                
                # Kronecker product: (batch_size, n_rx*n_tx, n_rx*n_tx)
                C = torch.einsum('bij,bkl->bikjl', C_tx, C_rx).reshape(
                    batch_size_actual, n_tx * n_rx, n_tx * n_rx)
            else:
                t_batch = torch.from_numpy(t[batch_indices]).to(dtype=torch_dtype, device=device)  # (batch_size, n_toep)
                C = self._toeplitz_batch(t_batch)  # (batch_size, n_antennas, n_antennas)
                C = C.transpose(-2, -1)  # Transpose as in original code
            
            # Compute CAh: (batch_size, n_antennas, n_antennas) @ (n_antennas, n_antennas) -> (batch_size, n_antennas, n_antennas)
            CAh = torch.einsum('bij,jk->bik', C, A_torch.conj().T)
            
            # Compute Cy: (batch_size, n_antennas, n_antennas)
            eye_matrix = torch.eye(A_torch.shape[0], dtype=torch_dtype, device=device).unsqueeze(0)
            Cy = torch.einsum('ij,bjk->bik', A_torch, CAh) + (1.0 / rho_torch) * eye_matrix
            
            # Solve linear system for all channels in batch
            # Cy: (batch_size, n_antennas, n_antennas)
            # y_batch: (batch_size, n_antennas)
            # We need to solve Cy @ x = y_batch for each batch
            y_batch_expanded = y_batch.unsqueeze(-1)  # (batch_size, n_antennas, 1)
            x = torch.linalg.solve(Cy, y_batch_expanded)  # (batch_size, n_antennas, 1)
            x = x.squeeze(-1)  # (batch_size, n_antennas)
            
            # Compute hest: (batch_size, n_antennas, n_antennas) @ (batch_size, n_antennas) -> (batch_size, n_antennas)
            hest_batch = torch.einsum('bij,bj->bi', CAh, x)
            
            hest_list.append(hest_batch.cpu().numpy())
        
        return np.concatenate(hest_list, axis=0)
    
    @staticmethod
    def _toeplitz_batch(c):
        """
        Build Toeplitz matrices in batch using PyTorch (vectorized).
        
        Args:
            c: torch tensor of shape (batch_size, n) - first column of Toeplitz matrices
            (device is inferred from c)
        
        Returns:
            torch tensor of shape (batch_size, n, n) - batch of Toeplitz matrices
        """
        batch_size, n = c.shape
        device = c.device
        r = c.conj()  # Hermitian Toeplitz: r = c.conj()
        
        # Create indices: diff[i,j] = i - j
        i_idx = torch.arange(n, device=device).view(1, n, 1)  # (1, n, 1)
        j_idx = torch.arange(n, device=device).view(1, 1, n)  # (1, 1, n)
        diff = i_idx - j_idx  # (1, n, n)
        
        # Create mask for positive and negative differences
        pos_mask = diff >= 0  # (1, n, n)
        
        # For positive diff: index into c, for negative diff: index into r
        # Clamp indices to valid range
        pos_idx = torch.clamp(diff, min=0, max=n-1).long()  # (1, n, n)
        neg_idx = torch.clamp(-diff, min=1, max=n-1).long()  # (1, n, n)
        
        # Use advanced indexing: for each (b, i, j), get c[b, pos_idx[i,j]] or r[b, neg_idx[i,j]]
        # Expand indices for batch indexing
        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(-1, n, n)  # (batch_size, n, n)
        pos_idx_expanded = pos_idx.expand(batch_size, -1, -1)  # (batch_size, n, n)
        neg_idx_expanded = neg_idx.expand(batch_size, -1, -1)  # (batch_size, n, n)
        
        # Index into c and r: c[batch_idx, pos_idx_expanded]
        pos_values = c[batch_idx, pos_idx_expanded]  # (batch_size, n, n)
        neg_values = r[batch_idx, neg_idx_expanded]  # (batch_size, n, n)
        
        # Combine using masks
        pos_mask_expanded = pos_mask.expand(batch_size, -1, -1)
        T = torch.where(pos_mask_expanded, pos_values, neg_values)
        
        return T
