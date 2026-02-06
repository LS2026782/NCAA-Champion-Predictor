"""
Game-by-game backtesting for NCAA Tournament predictions.

This module implements rolling-year cross-validation for game prediction:
- Train: all games from years < test_year
- Test: all games from test_year
- Evaluate: accuracy, log loss, upset prediction, by-round performance

TEMPORAL INTEGRITY:
- Strict separation: no future data leaks into training
- Skip 2020 (no tournament due to COVID)
- First test year depends on minimum training years required
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.game_loader import GameLoader
from src.models.game_predictor import GamePredictor
from config.settings import MIN_TRAIN_YEARS


class GameBacktester:
    """
    Backtests game prediction across multiple tournament years.
    
    Implements rolling-year cross-validation:
    - For each test_year, train on all games from prior years
    - Predict and evaluate all games in test_year
    - Aggregate results across all test years
    
    Attributes:
        game_loader: GameLoader instance
        results: List of per-year evaluation results
    """
    
    def __init__(self, model_type: str = 'logreg'):
        """
        Initialize the backtester.
        
        Args:
            model_type: 'logreg' or 'gbm'
        """
        self.model_type = model_type
        self.game_loader = GameLoader()
        self.results: List[Dict] = []
        self.all_predictions: Optional[pd.DataFrame] = None
        
    def run_backtest(
        self,
        first_test_year: Optional[int] = None,
        last_test_year: Optional[int] = None,
        min_train_games: int = 200,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run full backtest across all available years.
        
        Args:
            first_test_year: First year to test (default: earliest with enough training data)
            last_test_year: Last year to test (default: most recent)
            min_train_games: Minimum training games required
            verbose: Print progress
            
        Returns:
            DataFrame with results for each test year
        """
        # Load all data
        games_df, team_stats = self.game_loader.load_all()
        
        available_years = sorted(games_df['YEAR'].unique())
        
        # Determine test years
        if first_test_year is None:
            # Find first year with enough training data
            for i, year in enumerate(available_years):
                train_games = len(games_df[games_df['YEAR'] < year])
                if train_games >= min_train_games:
                    first_test_year = year
                    break
            if first_test_year is None:
                first_test_year = available_years[MIN_TRAIN_YEARS]
                
        if last_test_year is None:
            last_test_year = max(available_years)
            
        test_years = [y for y in available_years 
                      if y >= first_test_year and y <= last_test_year]
        
        if verbose:
            print("="*70)
            print("GAME-BY-GAME BACKTEST")
            print("="*70)
            print(f"Model: {self.model_type}")
            print(f"Test years: {test_years[0]} - {test_years[-1]} ({len(test_years)} tournaments)")
            print(f"Total games in dataset: {len(games_df)}")
            print()
        
        self.results = []
        all_preds = []
        
        for test_year in test_years:
            # Skip 2020 (COVID - no tournament)
            if test_year == 2020:
                continue
                
            if verbose:
                print(f"Testing {test_year}...", end=" ")
            
            # Split data
            train_games, test_games = self.game_loader.get_train_test_split(test_year)
            
            if len(train_games) < min_train_games:
                if verbose:
                    print(f"Skipped (only {len(train_games)} training games)")
                continue
            
            # Train model
            predictor = GamePredictor(model_type=self.model_type)
            predictor.fit(train_games, team_stats)
            
            # Predict and evaluate
            predictions = predictor.predict_games(test_games, team_stats)
            predictions['ACTUAL_WINNER'] = test_games['WINNER']
            predictions['CORRECT'] = (
                predictions['PREDICTED_WINNER'] == predictions['ACTUAL_WINNER']
            )
            
            # Calculate metrics
            metrics = self._calculate_year_metrics(predictions, test_year)
            self.results.append(metrics)
            
            # Store predictions
            predictions['TEST_YEAR'] = test_year
            all_preds.append(predictions)
            
            if verbose:
                print(f"Accuracy: {metrics['accuracy']:.1%} "
                      f"({metrics['correct']}/{metrics['total_games']} games)")
        
        # Combine all predictions
        self.all_predictions = pd.concat(all_preds, ignore_index=True)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        if verbose:
            self._print_summary(results_df)
            
        return results_df
    
    def _calculate_year_metrics(
        self, 
        predictions: pd.DataFrame,
        year: int
    ) -> Dict:
        """Calculate evaluation metrics for a single year."""
        correct = predictions['CORRECT'].sum()
        total = len(predictions)
        
        # Probability calibration
        probs = predictions['PROB_A'].values
        y_true = (predictions['ACTUAL_WINNER'] == predictions['TEAM_A']).astype(int)
        
        # Clip to avoid log(0)
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        log_loss_val = -np.mean(
            y_true * np.log(probs_clipped) + (1 - y_true) * np.log(1 - probs_clipped)
        )
        brier = np.mean((probs - y_true) ** 2)
        
        # By-round breakdown
        round_accuracy = {}
        for round_num in predictions['ROUND'].unique():
            round_preds = predictions[predictions['ROUND'] == round_num]
            round_accuracy[f'round_{int(round_num)}'] = round_preds['CORRECT'].mean()
        
        # Upset analysis
        potential_upsets = predictions[predictions['SEED_A'] < predictions['SEED_B']]
        upset_occurred = potential_upsets[
            potential_upsets['ACTUAL_WINNER'] == potential_upsets['TEAM_B']
        ]
        upset_predicted = potential_upsets[
            potential_upsets['PREDICTED_WINNER'] == potential_upsets['TEAM_B']
        ]
        
        # How many actual upsets did we predict?
        if len(upset_occurred) > 0:
            upset_ids = set(upset_occurred.index)
            predicted_ids = set(upset_predicted.index)
            upsets_correctly_predicted = len(upset_ids & predicted_ids)
        else:
            upsets_correctly_predicted = 0
        
        return {
            'year': year,
            'total_games': total,
            'correct': int(correct),
            'accuracy': correct / total,
            'log_loss': log_loss_val,
            'brier_score': brier,
            'upsets_occurred': len(upset_occurred),
            'upsets_predicted': len(upset_predicted),
            'upsets_correct': upsets_correctly_predicted,
            **round_accuracy
        }
    
    def _print_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary statistics."""
        print("\n" + "="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        
        total_games = results_df['total_games'].sum()
        total_correct = results_df['correct'].sum()
        
        print(f"\nOverall Accuracy: {total_correct/total_games:.1%} "
              f"({total_correct}/{total_games} games)")
        print(f"Mean Log Loss: {results_df['log_loss'].mean():.4f}")
        print(f"Mean Brier Score: {results_df['brier_score'].mean():.4f}")
        
        # Year-by-year
        print("\nYear-by-Year Results:")
        print("-" * 50)
        for _, row in results_df.iterrows():
            print(f"  {int(row['year'])}: {row['accuracy']:.1%} "
                  f"({int(row['correct'])}/{int(row['total_games'])} games) "
                  f"| Upsets: {int(row['upsets_correct'])}/{int(row['upsets_occurred'])}")
        
        # Upset summary
        print("\nUpset Prediction:")
        total_upsets = results_df['upsets_occurred'].sum()
        correctly_predicted = results_df['upsets_correct'].sum()
        total_predicted = results_df['upsets_predicted'].sum()
        
        print(f"  Actual upsets: {int(total_upsets)}")
        print(f"  Upsets we predicted: {int(total_predicted)}")
        print(f"  Upsets correctly predicted: {int(correctly_predicted)}")
        if total_upsets > 0:
            print(f"  Upset recall: {correctly_predicted/total_upsets:.1%}")
        if total_predicted > 0:
            print(f"  Upset precision: {correctly_predicted/total_predicted:.1%}")
        
        # By-round (if available)
        round_cols = [c for c in results_df.columns if c.startswith('round_')]
        if round_cols:
            print("\nAccuracy by Round (averaged across years):")
            for col in sorted(round_cols, key=lambda x: -int(x.split('_')[1])):
                round_num = col.split('_')[1]
                mean_acc = results_df[col].mean()
                round_name = {
                    '64': 'Round of 64',
                    '32': 'Round of 32', 
                    '16': 'Sweet 16',
                    '8': 'Elite 8',
                    '4': 'Final Four',
                    '2': 'Championship'
                }.get(round_num, f'Round {round_num}')
                print(f"  {round_name}: {mean_acc:.1%}")
    
    def get_predictions(self) -> pd.DataFrame:
        """Get all predictions from the backtest."""
        return self.all_predictions
    
    def get_wrong_predictions(self) -> pd.DataFrame:
        """Get games where the model was wrong."""
        if self.all_predictions is None:
            return pd.DataFrame()
        return self.all_predictions[~self.all_predictions['CORRECT']]
    
    def analyze_errors(self) -> pd.DataFrame:
        """Analyze patterns in wrong predictions."""
        wrong = self.get_wrong_predictions()
        
        if len(wrong) == 0:
            return pd.DataFrame()
        
        analysis = {
            'total_errors': len(wrong),
            'mean_seed_diff': abs(wrong['SEED_A'] - wrong['SEED_B']).mean(),
            'mean_confidence': wrong['PROB_A'].apply(
                lambda p: max(p, 1-p)
            ).mean(),
            'high_confidence_errors': (
                wrong['PROB_A'].apply(lambda p: max(p, 1-p)) > 0.7
            ).sum(),
            'upset_errors': (
                (wrong['SEED_A'] < wrong['SEED_B']) & 
                (wrong['ACTUAL_WINNER'] == wrong['TEAM_B'])
            ).sum(),
        }
        
        return pd.DataFrame([analysis])


def main():
    """Run full backtest."""
    backtester = GameBacktester(model_type='logreg')
    results = backtester.run_backtest(verbose=True)
    
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    error_analysis = backtester.analyze_errors()
    print(error_analysis.to_string())
    
    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results.to_csv(results_dir / 'game_backtest_results.csv', index=False)
    print(f"\nResults saved to {results_dir / 'game_backtest_results.csv'}")


if __name__ == "__main__":
    main()
