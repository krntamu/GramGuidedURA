"""
Lightweight diagnostic recorder for DPS evaluation.
Records c_t/b_t and clip_rate_cov statistics during sampling.
"""

from typing import Dict, List, Optional
import torch
from torch import Tensor


class DpsDiagnosticRecorder:
    """Records diagnostic statistics during DPS sampling."""
    
    def __init__(self, record_enabled: bool = True):
        self.record_enabled = record_enabled
        self.stats_history: List[Dict[str, float]] = []
        self.current_step_stats: Optional[Dict[str, float]] = None
    
    def start_step(self, t: int):
        """Start recording for a new timestep."""
        if not self.record_enabled:
            return
        self.current_step_stats = {'t': float(t)}
    
    def record_corrections(
        self,
        correction_like: Tensor,
        correction_cov: Tensor,
        beta_t: float,
    ):
        """Record correction magnitudes."""
        if not self.record_enabled or self.current_step_stats is None:
            return
        
        # Compute magnitudes (averaged over batch)
        norm_dims = tuple(range(1, correction_like.ndim))
        b_t = torch.linalg.vector_norm(correction_like, dim=norm_dims).mean().item()
        c_t = torch.linalg.vector_norm(correction_cov, dim=norm_dims).mean().item()
        
        self.current_step_stats['b_t'] = b_t
        self.current_step_stats['c_t'] = c_t
        self.current_step_stats['c_t / b_t'] = c_t / (b_t + 1e-10)
        self.current_step_stats['beta_t'] = beta_t
    
    def record_clip_rate(self, correction_cov: Tensor, correction_cov_clipped: Tensor):
        """Record clip rate for covariance correction."""
        if not self.record_enabled or self.current_step_stats is None:
            return
        
        # Check if any element was clipped
        diff = (correction_cov != correction_cov_clipped)
        diff_flat = diff.flatten(start_dim=1)
        clip_mask = diff_flat.any(dim=1)
        clip_rate = clip_mask.float().mean().item()
        
        self.current_step_stats['clip_rate_cov'] = clip_rate
    
    def finish_step(self):
        """Finish recording for current timestep."""
        if not self.record_enabled or self.current_step_stats is None:
            return
        self.stats_history.append(self.current_step_stats.copy())
        self.current_step_stats = None
    
    def get_late_stage_summary(self, t_start: float = 0.6, t_end: float = 0.9) -> Dict[str, float]:
        """
        Get summary statistics for mid-to-late stage.
        
        Parameters
        ----------
        t_start : float
            Start fraction of timesteps (0.6 = 60% through)
        t_end : float
            End fraction of timesteps (0.9 = 90% through)
        
        Returns
        -------
        summary : dict
            Mean statistics over the specified time window
        """
        if not self.stats_history:
            return {}
        
        total_steps = len(self.stats_history)
        late_start_idx = int(total_steps * t_start)
        late_end_idx = int(total_steps * t_end)
        if late_end_idx > total_steps:
            late_end_idx = total_steps
        
        late_stats = self.stats_history[late_start_idx:late_end_idx]
        if not late_stats:
            return {}
        
        summary = {
            'mean_c_over_b': sum(s.get('c_t / b_t', 0) for s in late_stats) / len(late_stats),
            'mean_clip_rate_cov': sum(s.get('clip_rate_cov', 0) for s in late_stats) / len(late_stats),
            'mean_c_t': sum(s.get('c_t', 0) for s in late_stats) / len(late_stats),
            'mean_b_t': sum(s.get('b_t', 0) for s in late_stats) / len(late_stats),
        }
        
        return summary
    
    def reset(self):
        """Reset all recorded statistics."""
        self.stats_history = []
        self.current_step_stats = None

