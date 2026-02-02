"""
Rolling-year backtesting for NCAA Championship Prediction.

This module implements strict temporal cross-validation:
- Train on all years < test_year
- Test on single year
- No leakage: future data never used in training

The backtester produces detailed results for each year and
aggregate statistics across all test years.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import MIN_YEAR, MAX_YEAR, MIN_TRAIN_YEARS, FIRST_TEST_YEAR
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.champion_model import ChampionPredictor


@dataclass
class YearResult:
    """Results for a single test year."""
    year: int
    champion_team: str
    champion_seed: int
    champion_prob: float
    champion_rank: int
    top_1_team: str
    top_1_prob: float
    top_5_teams: List[str]
    top_10_teams: List[str]
    brier_score: float
    log_loss: float
    all_predictions: pd.DataFrame
    

class Backtester:
    """
    Performs rolling-year backtesting for champion prediction.
    
    Implements strict temporal splitting to prevent leakage:
    - Train: years < test_year
    - Test: test_year only
    
    Tracks per-year metrics and aggregates across years.
    
    Attributes:
        loader: DataLoader instance
        builder: FeatureBuilder instance
        model_type: 'logreg' or 'gbm'
        results: List of YearResult objects
    """
    
    def __init__(
        self, 
        model_type: str = 'logreg',
        calibrate: bool = False
    ):
        """
        Initialize the backtester.
        
        Args:
            model_type: 'logreg' or 'gbm'
            calibrate: Whether to apply probability calibration
        """
        self.model_type = model_type
        self.calibrate = calibrate
        self.loader = DataLoader()
        self.builder = FeatureBuilder()
        self.results: List[YearResult] = []
        self._data_loaded = False
        
    def run_backtest(
        self, 
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        verbose: bool = True
    ) -> List[YearResult]:
        """
        Run full backtest across specified years.
        
        Args:
            start_year: First test year (default: FIRST_TEST_YEAR)
            end_year: Last test year (default: MAX_YEAR)
            verbose: Print progress
            
        Returns:
            List of YearResult objects
        """
        # Load data if not already loaded
        if not self._data_loaded:
            self.loader.load_all()
            self._data_loaded = True
            
        available_years = self.loader.get_years()
        
        # Determine test year range
        start_year = start_year or FIRST_TEST_YEAR
        end_year = end_year or MAX_YEAR
        
        test_years = [y for y in available_years if start_year <= y <= end_year]
        
        if verbose:
            print("="*70)
            print(f"BACKTEST: {self.model_type.upper()} | Years {start_year}-{end_year}")
            print("="*70)
            
        self.results = []
        
        for test_year in test_years:
            result = self._test_year(test_year, verbose)
            self.results.append(result)
            
        return self.results
    
    def _test_year(self, test_year: int, verbose: bool = True) -> YearResult:
        """
        Test model on a single year.
        
        Args:
            test_year: Year to test
            verbose: Print progress
            
        Returns:
            YearResult for this year
        """
        # Get train/test split
        train_df, test_df = self.loader.get_train_test_split(test_year)
        
        # Build features
        train_df_feat, X_train = self.builder.build_features(train_df, fit_scaler=True)
        test_df_feat, X_test = self.builder.build_features(test_df, fit_scaler=False)
        
        y_train = train_df_feat['IS_CHAMPION'].values
        y_test = test_df_feat['IS_CHAMPION'].values
        
        # Train model
        model = ChampionPredictor(model_type=self.model_type, calibrate=self.calibrate)
        model.fit(X_train, y_train, feature_names=self.builder.get_feature_names())
        
        # Predict
        probs = model.predict_proba(X_test)
        test_df_feat = test_df_feat.copy()
        test_df_feat['PROB'] = probs
        test_df_feat['RANK'] = test_df_feat['PROB'].rank(ascending=False).astype(int)
        
        # Get champion info
        champ_row = test_df_feat[test_df_feat['IS_CHAMPION'] == 1].iloc[0]
        champion_team = champ_row['TEAM']
        champion_seed = int(champ_row['SEED'])
        champion_prob = champ_row['PROB']
        champion_rank = int(champ_row['RANK'])
        
        # Get top predictions
        sorted_df = test_df_feat.sort_values('PROB', ascending=False)
        top_1_team = sorted_df.iloc[0]['TEAM']
        top_1_prob = sorted_df.iloc[0]['PROB']
        top_5_teams = sorted_df.head(5)['TEAM'].tolist()
        top_10_teams = sorted_df.head(10)['TEAM'].tolist()
        
        # Calculate metrics
        brier_score = np.mean((probs - y_test) ** 2)
        
        # Log loss with clipping to avoid log(0)
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        log_loss = -np.mean(y_test * np.log(probs_clipped) + 
                           (1 - y_test) * np.log(1 - probs_clipped))
        
        if verbose:
            in_top_5 = "Yes" if champion_rank <= 5 else "No"
            print(f"{test_year}: Champion={champion_team:20s} Seed={champion_seed} "
                  f"Rank={champion_rank:2d} Prob={champion_prob:.3f} Top5={in_top_5}")
            
        return YearResult(
            year=test_year,
            champion_team=champion_team,
            champion_seed=champion_seed,
            champion_prob=champion_prob,
            champion_rank=champion_rank,
            top_1_team=top_1_team,
            top_1_prob=top_1_prob,
            top_5_teams=top_5_teams,
            top_10_teams=top_10_teams,
            brier_score=brier_score,
            log_loss=log_loss,
            all_predictions=test_df_feat[['TEAM', 'SEED', 'PROB', 'RANK', 'IS_CHAMPION']].copy()
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get aggregate summary statistics across all backtest years.
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.results:
            raise ValueError("No results. Run backtest first.")
            
        ranks = [r.champion_rank for r in self.results]
        probs = [r.champion_prob for r in self.results]
        brier_scores = [r.brier_score for r in self.results]
        log_losses = [r.log_loss for r in self.results]
        
        summary = {
            'model_type': self.model_type,
            'n_years': len(self.results),
            'years_tested': [r.year for r in self.results],
            
            # Champion rank statistics
            'mean_champion_rank': np.mean(ranks),
            'median_champion_rank': np.median(ranks),
            'std_champion_rank': np.std(ranks),
            'min_champion_rank': min(ranks),
            'max_champion_rank': max(ranks),
            
            # Top-K inclusion rates
            'top_1_rate': np.mean([r <= 1 for r in ranks]),
            'top_5_rate': np.mean([r <= 5 for r in ranks]),
            'top_10_rate': np.mean([r <= 10 for r in ranks]),
            'top_25_rate': np.mean([r <= 25 for r in ranks]),
            
            # Probability statistics
            'mean_champion_prob': np.mean(probs),
            'min_champion_prob': min(probs),
            'max_champion_prob': max(probs),
            
            # Error metrics
            'mean_brier_score': np.mean(brier_scores),
            'mean_log_loss': np.mean(log_losses),
            
            # Per-year breakdown
            'year_details': [
                {
                    'year': r.year,
                    'champion': r.champion_team,
                    'seed': r.champion_seed,
                    'rank': r.champion_rank,
                    'prob': r.champion_prob,
                    'top_1': r.top_1_team,
                    'in_top_5': r.champion_rank <= 5
                }
                for r in self.results
            ]
        }
        
        return summary
    
    def print_report(self) -> None:
        """Print a formatted backtest report."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print(f"BACKTEST RESULTS: {summary['model_type'].upper()}")
        print(f"Test Years: {summary['years_tested'][0]} - {summary['years_tested'][-1]} "
              f"({summary['n_years']} years)")
        print("="*70)
        
        print("\n--- Champion Rank Statistics ---")
        print(f"  Mean Rank:   {summary['mean_champion_rank']:.1f}")
        print(f"  Median Rank: {summary['median_champion_rank']:.1f}")
        print(f"  Std Dev:     {summary['std_champion_rank']:.1f}")
        print(f"  Best Rank:   {summary['min_champion_rank']}")
        print(f"  Worst Rank:  {summary['max_champion_rank']}")
        
        print("\n--- Top-K Inclusion Rates ---")
        print(f"  Top-1:  {summary['top_1_rate']*100:5.1f}% ({int(summary['top_1_rate']*summary['n_years'])}/{summary['n_years']})")
        print(f"  Top-5:  {summary['top_5_rate']*100:5.1f}% ({int(summary['top_5_rate']*summary['n_years'])}/{summary['n_years']})")
        print(f"  Top-10: {summary['top_10_rate']*100:5.1f}% ({int(summary['top_10_rate']*summary['n_years'])}/{summary['n_years']})")
        print(f"  Top-25: {summary['top_25_rate']*100:5.1f}% ({int(summary['top_25_rate']*summary['n_years'])}/{summary['n_years']})")
        
        print("\n--- Champion Probability Stats ---")
        print(f"  Mean Prob:   {summary['mean_champion_prob']:.3f}")
        print(f"  Min Prob:    {summary['min_champion_prob']:.3f}")
        print(f"  Max Prob:    {summary['max_champion_prob']:.3f}")
        
        print("\n--- Error Metrics ---")
        print(f"  Mean Brier Score: {summary['mean_brier_score']:.5f}")
        print(f"  Mean Log Loss:    {summary['mean_log_loss']:.4f}")
        
        print("\n--- Year-by-Year Results ---")
        print(f"{'Year':<6} {'Champion':<20} {'Seed':<5} {'Rank':<5} {'Prob':<7} {'Top-1 Predicted':<20}")
        print("-"*70)
        for detail in summary['year_details']:
            marker = "*" if detail['rank'] == 1 else ""
            print(f"{detail['year']:<6} {detail['champion']:<20} {detail['seed']:<5} "
                  f"{detail['rank']:<5} {detail['prob']:<7.3f} {detail['top_1']:<20} {marker}")
        
        # Identify hardest years
        print("\n--- Hardest Years (Champion ranked > 10) ---")
        hard_years = [d for d in summary['year_details'] if d['rank'] > 10]
        if hard_years:
            for d in hard_years:
                print(f"  {d['year']}: {d['champion']} (Seed {d['seed']}) - Ranked {d['rank']}")
        else:
            print("  None! All champions ranked in top 10.")
            
    def save_results(self, filepath: str) -> None:
        """
        Save backtest results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        summary = self.get_summary()
        summary['timestamp'] = datetime.now().isoformat()
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        summary = convert_types(summary)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Results saved to {filepath}")


def main():
    """Run full backtest."""
    print("Running full backtest with Logistic Regression...")
    
    backtester = Backtester(model_type='logreg')
    backtester.run_backtest()
    backtester.print_report()
    
    # Save results
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    backtester.save_results(str(results_dir / "backtest_logreg.json"))


if __name__ == "__main__":
    main()
