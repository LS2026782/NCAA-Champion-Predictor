"""
Evaluation metrics for NCAA Championship Prediction.

This module provides:
- Standard classification metrics (Brier score, log loss)
- Champion-specific metrics (rank, top-K inclusion)
- Calibration analysis
- Visualization utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class EvaluationMetrics:
    """
    Computes evaluation metrics for champion prediction.
    
    Metrics computed:
    - Brier Score: measures probability calibration
    - Log Loss: penalizes confident wrong predictions
    - AUC-ROC: discrimination ability
    - Champion Rank: where actual champion falls in ranking
    - Top-K Inclusion: whether champion was in top K predictions
    - Calibration Curve: reliability diagram data
    """
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        team_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True binary labels (1 = champion)
            y_prob: Predicted probabilities
            team_names: Optional team names for ranking
            
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        # Brier Score (lower is better)
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        
        # Log Loss (lower is better)
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        metrics['log_loss'] = log_loss(y_true, y_prob_clipped)
        
        # AUC-ROC (higher is better) - only if both classes present
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['auc_roc'] = np.nan
            
        # Champion rank
        if y_true.sum() > 0:
            champion_idx = np.where(y_true == 1)[0][0]
            ranks = (-y_prob).argsort().argsort() + 1  # 1-indexed ranks
            metrics['champion_rank'] = int(ranks[champion_idx])
            metrics['champion_prob'] = float(y_prob[champion_idx])
        else:
            metrics['champion_rank'] = np.nan
            metrics['champion_prob'] = np.nan
            
        # Top-K inclusion
        for k in [1, 5, 10, 25]:
            if not np.isnan(metrics['champion_rank']):
                metrics[f'in_top_{k}'] = int(metrics['champion_rank'] <= k)
            else:
                metrics[f'in_top_{k}'] = np.nan
                
        return metrics
    
    @staticmethod
    def champion_rank(y_true: np.ndarray, y_prob: np.ndarray) -> int:
        """
        Get the rank of the actual champion by predicted probability.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Rank of champion (1 = highest probability)
        """
        champion_idx = np.where(y_true == 1)[0][0]
        # Rank from highest prob (1) to lowest
        ranks = (-y_prob).argsort().argsort() + 1
        return int(ranks[champion_idx])
    
    @staticmethod
    def top_k_inclusion(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        k: int
    ) -> bool:
        """
        Check if champion is in top-K predictions.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities  
            k: Number of top predictions to consider
            
        Returns:
            True if champion is in top K
        """
        rank = EvaluationMetrics.champion_rank(y_true, y_prob)
        return rank <= k
    
    @staticmethod
    def calibration_data(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve data.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Tuple of (fraction_of_positives, mean_predicted_value)
        """
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy='uniform'
            )
            return fraction_of_positives, mean_predicted_value
        except ValueError:
            # Not enough data for calibration curve
            return np.array([]), np.array([])
    
    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE is the weighted average of |accuracy - confidence| across bins.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            ECE value (lower is better)
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                bin_weight = mask.sum() / len(y_prob)
                ece += bin_weight * np.abs(bin_acc - bin_conf)
                
        return ece
    
    @staticmethod
    def summarize_backtest_results(
        year_results: List[Dict]
    ) -> pd.DataFrame:
        """
        Create summary DataFrame from backtest results.
        
        Args:
            year_results: List of per-year result dictionaries
            
        Returns:
            DataFrame with summary statistics
        """
        df = pd.DataFrame(year_results)
        
        summary = pd.DataFrame({
            'Metric': [
                'Mean Champion Rank',
                'Median Champion Rank',
                'Top-1 Rate',
                'Top-5 Rate', 
                'Top-10 Rate',
                'Top-25 Rate',
                'Mean Champion Prob',
                'Mean Brier Score',
                'Mean Log Loss'
            ],
            'Value': [
                df['champion_rank'].mean(),
                df['champion_rank'].median(),
                (df['champion_rank'] <= 1).mean(),
                (df['champion_rank'] <= 5).mean(),
                (df['champion_rank'] <= 10).mean(),
                (df['champion_rank'] <= 25).mean(),
                df['champion_prob'].mean(),
                df['brier_score'].mean(),
                df['log_loss'].mean()
            ]
        })
        
        return summary


def main():
    """Test evaluation metrics."""
    # Create sample data
    np.random.seed(42)
    n_teams = 68
    
    # Simulate probabilities (one champion)
    y_true = np.zeros(n_teams)
    y_true[0] = 1  # First team is champion
    
    # Simulate probabilities where champion has high prob
    y_prob = np.random.beta(2, 5, n_teams)
    y_prob[0] = 0.15  # Champion has 15% prob
    y_prob = y_prob / y_prob.sum()  # Normalize (optional)
    
    print("="*60)
    print("METRICS TEST")
    print("="*60)
    
    metrics = EvaluationMetrics.compute_all_metrics(y_true, y_prob)
    
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")
            
    # Test calibration
    print("\nCalibration Error:")
    ece = EvaluationMetrics.expected_calibration_error(y_true, y_prob)
    print(f"  ECE: {ece:.4f}")


if __name__ == "__main__":
    main()
