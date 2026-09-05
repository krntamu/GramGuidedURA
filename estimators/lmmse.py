import numpy as np
import torch
from modules.utils import toeplitz


def mp_eval(obj, y, toep, genie, A=None, is_independent=False):
    if genie:
        if is_independent:
            hest = obj.estimate_genie_independent_cols(y, toep, A)
        else:
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

    def estimate_genie_independent_cols(self, y, t, A=None):
        """
        CPU LMMSE for independent columns (e.g., multi-user scenario).
        y: shape [n_batches, n_rx * n_tx]
        t: tuple (t_rx, t_tx) where t_rx is [n_batches, n_tx, n_rx]
        """
        (n_batches, n_antennas) = y.shape
        t_rx, _ = t
        n_tx = t_rx.shape[1]
        n_rx = t_rx.shape[2]
        
        if A is not None:
            raise NotImplementedError("A matrix not implemented for independent cols yet")
            
        hest = np.zeros([n_batches, n_antennas], dtype=y.dtype)
        y_reshaped = y.reshape(n_batches, n_tx, n_rx)
        
        for b in range(n_batches):
            for k in range(n_tx):
                C_rx = toeplitz(t_rx[b, k, :])
                Cy = C_rx + 1 / self.rho * np.eye(n_rx)
                y_col = y_reshaped[b, k, :]
                hest_col = C_rx @ np.linalg.solve(Cy, y_col)
                hest[b, k * n_rx : (k + 1) * n_rx] = hest_col
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
        """
        import torch
        
        n_batches, n_antennas = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=y.dtype)
            
        # Determine dtype
        if y.dtype == np.complex128:
            torch_dtype = torch.complex128
        elif y.dtype == np.complex64:
            torch_dtype = torch.complex64
        else:
            torch_dtype = torch.complex128
            
        # Add epsilon to diagonal to prevent singular matrix errors in solve()
        # This is especially important for multi-user cases with high condition numbers
        epsilon = 1e-6
        
        y_torch = torch.from_numpy(y).to(dtype=torch_dtype, device=device)
        A_torch = torch.from_numpy(A).to(dtype=torch_dtype, device=device)
        rho_torch = torch.tensor(self.rho, dtype=torch_dtype, device=device)
        
        hest_list = []
        
        for batch_start in range(0, n_batches, batch_size):
            batch_end = min(batch_start + batch_size, n_batches)
            batch_indices = slice(batch_start, batch_end)
            batch_size_actual = batch_end - batch_start
            
            y_batch = y_torch[batch_indices]
            
            if type(t) is tuple:
                t_rx, t_tx = t
                t_rx_batch = torch.from_numpy(t_rx[batch_indices]).to(dtype=torch_dtype, device=device)
                t_tx_batch = torch.from_numpy(t_tx[batch_indices]).to(dtype=torch_dtype, device=device)
                
                n_rx = t_rx_batch.shape[1]
                n_tx = t_tx_batch.shape[1]
                
                C_rx = self._toeplitz_batch(t_rx_batch)
                C_tx = self._toeplitz_batch(t_tx_batch)
                
                C = torch.einsum('bij,bkl->bikjl', C_tx, C_rx).reshape(
                    batch_size_actual, n_tx * n_rx, n_tx * n_rx)
            else:
                t_batch = torch.from_numpy(t[batch_indices]).to(dtype=torch_dtype, device=device)
                C = self._toeplitz_batch(t_batch)
                C = C.transpose(-2, -1)
            
            CAh = torch.einsum('bij,jk->bik', C, A_torch.conj().T)
            
            eye_matrix = torch.eye(A_torch.shape[0], dtype=torch_dtype, device=device).unsqueeze(0)
            # Add epsilon stabilizer to the diagonal of Cy
            Cy = torch.einsum('ij,bjk->bik', A_torch, CAh) + (1.0 / rho_torch + epsilon) * eye_matrix
            
            y_batch_expanded = y_batch.unsqueeze(-1)
            try:
                x = torch.linalg.solve(Cy, y_batch_expanded)
            except torch.linalg.LinAlgError:
                # Fallback to pseudo-inverse if solve fails even with epsilon
                print(f"Warning: torch.linalg.solve failed for batch {batch_start}, falling back to pinv")
                Cy_inv = torch.linalg.pinv(Cy, hermitian=True)
                x = torch.einsum('bij,bjk->bik', Cy_inv, y_batch_expanded)
                
            x = x.squeeze(-1)
            
            hest_batch = torch.einsum('bij,bj->bi', CAh, x)
            hest_list.append(hest_batch.cpu().numpy())
            
        return np.concatenate(hest_list, axis=0)
    
    def estimate_genie_independent_cols_gpu(self, y, t, device='cuda', batch_size=100):
        """
        GPU-accelerated LMMSE for independent columns (e.g., multi-user scenario).
        y: shape [n_batches, n_rx * n_tx]
        t: tuple (t_rx, t_tx) where t_rx is [n_batches, n_tx, n_rx] and t_tx is [n_batches, n_tx]
           Since columns are independent, t_tx is actually just variance, and we can ignore it 
           if we assume variance is 1, or apply it. The pseudo-multiuser dataset creates 
           identity Tx cov, so we just use t_rx for each column.
        """
        import torch
        
        n_batches, n_antennas = y.shape
        # Assuming n_antennas = n_rx * n_tx
        t_rx, _ = t
        n_tx = t_rx.shape[1]
        n_rx = t_rx.shape[2]
        
        if y.dtype == np.complex128:
            torch_dtype = torch.complex128
        elif y.dtype == np.complex64:
            torch_dtype = torch.complex64
        else:
            torch_dtype = torch.complex128
            
        epsilon = 1e-6
        y_torch = torch.from_numpy(y).to(dtype=torch_dtype, device=device)
        # Reshape y to [n_batches, n_tx, n_rx] (assuming Fortran order or C order?)
        # Wait, in baselines.py: channels_test = np.reshape(channels_test, (-1, n_antennas), 'F')
        # This means the 1024 elements are [rx1_tx1, rx2_tx1, ..., rx64_tx1, rx1_tx2, ...]
        # So we can reshape to [n_batches, n_tx, n_rx] by view(n_batches, n_tx, n_rx)
        y_torch = y_torch.view(n_batches, n_tx, n_rx)
        
        rho_torch = torch.tensor(self.rho, dtype=torch_dtype, device=device)
        eye_matrix = torch.eye(n_rx, dtype=torch_dtype, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, n_rx, n_rx]
        
        hest_list = []
        
        for batch_start in range(0, n_batches, batch_size):
            batch_end = min(batch_start + batch_size, n_batches)
            batch_indices = slice(batch_start, batch_end)
            batch_size_actual = batch_end - batch_start
            
            y_batch = y_torch[batch_indices] # [batch, n_tx, n_rx]
            t_rx_batch = torch.from_numpy(t_rx[batch_indices]).to(dtype=torch_dtype, device=device) # [batch, n_tx, n_rx]
            
            # Flatten batch and n_tx to compute toeplitz matrices efficiently
            t_rx_flat = t_rx_batch.reshape(batch_size_actual * n_tx, n_rx)
            C_rx_flat = self._toeplitz_batch(t_rx_flat) # [batch * n_tx, n_rx, n_rx]
            C_rx = C_rx_flat.view(batch_size_actual, n_tx, n_rx, n_rx)
            
            # Cy = C_rx + (1/rho + epsilon) * I
            Cy = C_rx + (1.0 / rho_torch + epsilon) * eye_matrix
            
            y_batch_expanded = y_batch.unsqueeze(-1) # [batch, n_tx, n_rx, 1]
            try:
                # Solve linear system for each column independently
                x = torch.linalg.solve(Cy, y_batch_expanded) # [batch, n_tx, n_rx, 1]
            except torch.linalg.LinAlgError:
                print(f"Warning: torch.linalg.solve failed for batch {batch_start}, falling back to pinv")
                Cy_inv = torch.linalg.pinv(Cy, hermitian=True)
                x = torch.einsum('btij,btjk->btik', Cy_inv, y_batch_expanded)
                
            x = x.squeeze(-1) # [batch, n_tx, n_rx]
            
            # hest = C_rx @ x
            hest_batch = torch.einsum('btij,btj->bti', C_rx, x) # [batch, n_tx, n_rx]
            
            # Flatten back to [batch, n_antennas]
            hest_batch_flat = hest_batch.view(batch_size_actual, -1)
            hest_list.append(hest_batch_flat.cpu().numpy())
            
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
