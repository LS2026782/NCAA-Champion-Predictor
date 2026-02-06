"""
Game-by-game prediction script for NCAA Tournament.

This script predicts individual game outcomes for a tournament year.
Can be used for:
- Backtesting a specific year
- Predicting upcoming tournament games (2025)
- Comparing to other prediction methods

Usage:
    python predict_games.py                    # Backtest latest year (2024)
    python predict_games.py --year 2023        # Backtest 2023
    python predict_games.py --year 2025        # Predict 2025 (no scores yet)
    python predict_games.py --backtest         # Full backtest all years
    python predict_games.py --model gbm        # Use gradient boosting
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data.game_loader import GameLoader
from src.models.game_predictor import GamePredictor
from src.evaluation.game_backtester import GameBacktester


def predict_year(year: int, model_type: str = 'logreg', verbose: bool = True):
    """
    Predict all games for a specific tournament year.
    
    Args:
        year: Tournament year to predict
        model_type: 'logreg' or 'gbm'
        verbose: Print detailed output
    """
    print("="*70)
    print(f"GAME-BY-GAME PREDICTIONS FOR {year}")
    print("="*70)
    
    # Load data
    loader = GameLoader()
    games_df, team_stats = loader.load_all()
    
    # Check if year is in dataset
    available_years = sorted(games_df['YEAR'].unique())
    
    if year not in available_years and year != 2025:
        print(f"Error: Year {year} not found. Available: {available_years}")
        return None
    
    # Get train/test split
    train_games = games_df[games_df['YEAR'] < year]
    
    if year in available_years:
        test_games = games_df[games_df['YEAR'] == year]
        has_results = True
    else:
        # For future years (2025), load from matchups file
        print(f"Year {year} has no results yet. Loading bracket structure...")
        from config.settings import TOURNAMENT_MATCHUPS_FILE
        matchups = pd.read_csv(TOURNAMENT_MATCHUPS_FILE)
        year_matchups = matchups[matchups['YEAR'] == year]
        
        if len(year_matchups) == 0:
            print(f"No bracket data found for {year}")
            return None
            
        # For 2025, just show team rankings by predicted strength
        print(f"\nFound {len(year_matchups)} teams in {year} bracket")
        has_results = False
        test_games = None
    
    print(f"\nTraining on {len(train_games)} games ({train_games['YEAR'].min()}-{train_games['YEAR'].max()})")
    
    # Train model
    predictor = GamePredictor(model_type=model_type)
    predictor.fit(train_games, team_stats)
    
    # Feature importance
    if verbose:
        print("\nTop Features for Game Prediction:")
        importance = predictor.get_feature_importance()
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']:15s}: {row['importance']:.4f}")
    
    if has_results:
        # Predict and evaluate
        predictions = predictor.predict_games(test_games, team_stats)
        predictions['ACTUAL_WINNER'] = test_games['WINNER']
        predictions['CORRECT'] = predictions['PREDICTED_WINNER'] == predictions['ACTUAL_WINNER']
        
        # Calculate accuracy
        accuracy = predictions['CORRECT'].mean()
        
        print(f"\n{'='*70}")
        print(f"RESULTS FOR {year}")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {accuracy:.1%} ({predictions['CORRECT'].sum()}/{len(predictions)})")
        
        # By round
        print("\nBy Round:")
        for round_num in sorted(predictions['ROUND'].unique(), reverse=True):
            round_preds = predictions[predictions['ROUND'] == round_num]
            round_acc = round_preds['CORRECT'].mean()
            round_name = {
                64: 'Round of 64',
                32: 'Round of 32',
                16: 'Sweet 16',
                8: 'Elite 8',
                4: 'Final Four',
                2: 'Championship'
            }.get(round_num, f'Round {round_num}')
            print(f"  {round_name}: {round_acc:.1%} ({round_preds['CORRECT'].sum()}/{len(round_preds)})")
        
        # Show incorrect predictions
        if verbose:
            wrong = predictions[~predictions['CORRECT']].copy()
            print(f"\nIncorrect Predictions ({len(wrong)} games):")
            print("-"*70)
            
            # Sort by round (later rounds first)
            wrong = wrong.sort_values('ROUND')
            
            for _, row in wrong.iterrows():
                round_name = {64: 'R64', 32: 'R32', 16: 'S16', 8: 'E8', 4: 'F4', 2: 'Champ'}.get(row['ROUND'], '')
                prob = row['PROB_A'] if row['PREDICTED_WINNER'] == row['TEAM_A'] else 1 - row['PROB_A']
                print(f"  {round_name}: Predicted {row['PREDICTED_WINNER']:20s} ({prob:.1%}) | "
                      f"Actual: {row['ACTUAL_WINNER']}")
        
        # Save predictions
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        predictions.to_csv(results_dir / f'game_predictions_{year}.csv', index=False)
        print(f"\nPredictions saved to results/game_predictions_{year}.csv")
        
        return predictions
    
    else:
        # For 2025 - show team power rankings
        year_stats = team_stats[team_stats['YEAR'] == year].copy()
        
        if len(year_stats) == 0:
            print(f"No team stats found for {year}")
            return None
        
        print(f"\n{year} Tournament Teams Power Rankings:")
        print("-"*50)
        
        # Rank by KADJ EM
        year_stats = year_stats.sort_values('KADJ EM', ascending=False)
        
        for i, (_, row) in enumerate(year_stats.head(25).iterrows(), 1):
            seed = int(row['SEED_NUM']) if pd.notna(row['SEED_NUM']) else '?'
            print(f"  {i:2d}. ({seed:2}) {row['TEAM']:20s} | EM: {row['KADJ EM']:+6.2f}")
        
        return year_stats


def run_full_backtest(model_type: str = 'logreg'):
    """Run full backtest across all years."""
    backtester = GameBacktester(model_type=model_type)
    results = backtester.run_backtest(verbose=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='NCAA Tournament Game-by-Game Prediction'
    )
    parser.add_argument(
        '--year', 
        type=int, 
        default=2024,
        help='Tournament year to predict (default: 2024)'
    )
    parser.add_argument(
        '--model',
        choices=['logreg', 'gbm'],
        default='logreg',
        help='Model type: logreg or gbm (default: logreg)'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run full backtest across all years'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    if args.backtest:
        run_full_backtest(model_type=args.model)
    else:
        predict_year(
            year=args.year,
            model_type=args.model,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
