"""
This module provides functions for collecting and analyzing verification metrics.
"""

import logging
import time
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class VerificationMetrics:
    """
    A class for collecting and analyzing verification metrics.
    """
    
    def __init__(self):
        """
        Initialize the VerificationMetrics.
        """
        self.metrics = {
            'start_time': time.time(),
            'verification_times': [],
            'comparison_times': [],
            'scores': [],
            'top_indices': [],
            'win_counts': [],
        }
    
    def record_verification(self, scores: List[float], verification_time: float):
        """
        Record verification metrics.
        
        Args:
            scores: The verification scores.
            verification_time: The time taken for verification.
        """
        self.metrics['scores'] = scores
        self.metrics['verification_times'].append(verification_time)
        
        if scores:
            self.metrics['max_score'] = max(scores)
            self.metrics['min_score'] = min(scores)
            self.metrics['avg_score'] = sum(scores) / len(scores)
        
        logger.info(f"Recorded verification metrics: {len(scores)} scores, {verification_time:.2f}s")
    
    def record_comparison(self, top_indices: List[int], win_counts: List[int], comparison_time: float):
        """
        Record comparison metrics.
        
        Args:
            top_indices: The indices of the top responses.
            win_counts: The win counts for each response.
            comparison_time: The time taken for comparison.
        """
        self.metrics['top_indices'] = top_indices
        self.metrics['win_counts'] = win_counts
        self.metrics['comparison_times'].append(comparison_time)
        
        logger.info(f"Recorded comparison metrics: {len(top_indices)} top indices, {comparison_time:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the verification metrics.
        
        Returns:
            A dictionary containing the verification metrics summary.
        """
        total_time = time.time() - self.metrics['start_time']
        
        summary = {
            'total_time': total_time,
            'verification_time': sum(self.metrics['verification_times']),
            'comparison_time': sum(self.metrics['comparison_times']),
            'num_responses': len(self.metrics.get('scores', [])),
            'num_top_responses': len(self.metrics.get('top_indices', [])),
        }
        
        if 'max_score' in self.metrics:
            summary['max_score'] = self.metrics['max_score']
            summary['min_score'] = self.metrics['min_score']
            summary['avg_score'] = self.metrics['avg_score']
        
        logger.info(f"Verification metrics summary: {summary}")
        return summary
