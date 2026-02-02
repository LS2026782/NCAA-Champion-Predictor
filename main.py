"""
NCAA Championship Prediction Pipeline - Main Entry Point

This script orchestrates the full prediction pipeline:
1. Load and preprocess data
2. Train champion prediction models
3. Run historical backtesting
4. Generate predictions for current/upcoming tournament
5. Run Monte Carlo simulations
6. Generate comprehensive reports

Usage:
    python main.py                    # Run full pipeline
    python main.py --backtest         # Run backtest only
    python main.py --predict 2024     # Predict for specific year
    python main.py --simulate 2024    # Run Monte Carlo for year
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    MIN_YEAR, MAX_YEAR, FIRST_TEST_YEAR, 
    RESULTS_DIR, MODEL_FEATURES
)
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.champion_model import ChampionPredictor
from src.evaluation.backtester import Backtester
from src.simulation.monte_carlo import BracketSimulator


def run_backtest(model_type: str = 'logreg', save: bool = True) -> dict:
    """
    Run full historical backtest.
    
    Args:
        model_type: 'logreg' or 'gbm'
        save: Whether to save results to file
        
    Returns:
        Summary dictionary
    """
    print("\n" + "="*70)
    print("RUNNING HISTORICAL BACKTEST")
    print("="*70)
    
    backtester = Backtester(model_type=model_type)
    backtester.run_backtest()
    backtester.print_report()
    
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        filepath = RESULTS_DIR / f"backtest_{model_type}_{datetime.now().strftime('%Y%m%d')}.json"
        backtester.save_results(str(filepath))
        
    return backtester.get_summary()


def predict_year(year: int, model_type: str = 'logreg') -> dict:
    """
    Generate predictions for a specific year.
    
    Args:
        year: Tournament year to predict
        model_type: Model type to use
        
    Returns:
        Dictionary with predictions
    """
    print("\n" + "="*70)
    print(f"GENERATING PREDICTIONS FOR {year}")
    print("="*70)
    
    # Load data
    loader = DataLoader()
    loader.load_all()
    
    # Check if year is available
    available_years = loader.get_years()
    if year not in available_years:
        print(f"Error: Year {year} not in available years: {available_years}")
        return {}
    
    # Get train/test split
    train_df, test_df = loader.get_train_test_split(year)
    
    # Build features
    builder = FeatureBuilder()
    train_df_feat, X_train = builder.build_features(train_df, fit_scaler=True)
    test_df_feat, X_test = builder.build_features(test_df, fit_scaler=False)
    
    y_train = train_df_feat['IS_CHAMPION'].values
    
    # Train model
    print(f"\nTraining {model_type.upper()} model...")
    model = ChampionPredictor(model_type=model_type)
    model.fit(X_train, y_train, feature_names=builder.get_feature_names())
    
    # Predict
    probs = model.predict_proba(X_test)
    test_df_feat = test_df_feat.copy()
    test_df_feat['PROB'] = probs
    test_df_feat['RANK'] = test_df_feat['PROB'].rank(ascending=False).astype(int)
    
    # Sort by probability
    predictions = test_df_feat.sort_values('PROB', ascending=False)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TOP 25 CHAMPIONSHIP PREDICTIONS FOR {year}")
    print(f"{'='*60}")
    print(f"{'Rank':<6}{'Team':<25}{'Seed':<6}{'Prob':<10}{'Actual':<10}")
    print("-"*60)
    
    for _, row in predictions.head(25).iterrows():
        is_champ = "CHAMPION" if row['IS_CHAMPION'] == 1 else ""
        print(f"{int(row['RANK']):<6}{row['TEAM']:<25}{int(row['SEED']):<6}"
              f"{row['PROB']:.3f}     {is_champ}")
    
    # Feature importance
    if model_type == 'logreg':
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE (Logistic Regression Coefficients)")
        print(f"{'='*60}")
        coefs = model.get_coefficients()
        print(coefs[['feature', 'coefficient', 'odds_ratio']].to_string(index=False))
    
    # Save predictions
    RESULTS_DIR.mkdir(exist_ok=True)
    predictions_file = RESULTS_DIR / f"predictions_{year}_{model_type}.csv"
    predictions[['RANK', 'TEAM', 'SEED', 'PROB', 'IS_CHAMPION']].to_csv(
        predictions_file, index=False
    )
    print(f"\nPredictions saved to {predictions_file}")
    
    return {
        'year': year,
        'model_type': model_type,
        'top_5': predictions.head(5)[['TEAM', 'SEED', 'PROB']].to_dict('records'),
        'champion_rank': int(predictions[predictions['IS_CHAMPION']==1]['RANK'].values[0])
        if predictions['IS_CHAMPION'].sum() > 0 else None
    }


def run_simulation(year: int, n_simulations: int = 10000) -> dict:
    """
    Run Monte Carlo bracket simulation for a year.
    
    Args:
        year: Tournament year
        n_simulations: Number of simulations
        
    Returns:
        Dictionary with simulation results
    """
    print("\n" + "="*70)
    print(f"RUNNING MONTE CARLO SIMULATION FOR {year}")
    print(f"Simulations: {n_simulations:,}")
    print("="*70)
    
    # Load data
    loader = DataLoader()
    loader.load_all()
    
    train_df, test_df = loader.get_train_test_split(year)
    
    # Build features
    builder = FeatureBuilder()
    train_df_feat, _ = builder.build_features(train_df, fit_scaler=True)
    test_df_feat, _ = builder.build_features(test_df, fit_scaler=False)
    
    # Create and fit simulator
    print("\nFitting game-level model...")
    simulator = BracketSimulator(n_simulations=n_simulations)
    simulator.fit(train_df_feat)
    
    # Run simulation
    print("Running simulations...")
    sim_probs = simulator.simulate_tournament(test_df_feat, verbose=True)
    
    # Sort results
    sorted_probs = sorted(sim_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"MONTE CARLO CHAMPIONSHIP ODDS FOR {year}")
    print(f"{'='*60}")
    
    for rank, (team, prob) in enumerate(sorted_probs[:20], 1):
        seed = test_df_feat[test_df_feat['TEAM'] == team]['SEED'].values[0]
        is_champ = test_df_feat[test_df_feat['TEAM'] == team]['IS_CHAMPION'].values[0]
        marker = " <-- CHAMPION" if is_champ else ""
        print(f"{rank:3d}. {team:<25} (Seed {int(seed):2d}): {prob:6.2%}{marker}")
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    sim_file = RESULTS_DIR / f"simulation_{year}.json"
    with open(sim_file, 'w') as f:
        json.dump({
            'year': year,
            'n_simulations': n_simulations,
            'championship_odds': {k: float(v) for k, v in sorted_probs}
        }, f, indent=2)
    print(f"\nSimulation results saved to {sim_file}")
    
    return {
        'year': year,
        'n_simulations': n_simulations,
        'top_5': sorted_probs[:5]
    }


def run_full_pipeline(year: int = None):
    """
    Run the complete prediction pipeline.
    
    Args:
        year: Year to predict (default: most recent)
    """
    print("\n" + "="*70)
    print("NCAA CHAMPIONSHIP PREDICTION PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Run backtest with logistic regression
    print("\n[1/4] Running historical backtest (Logistic Regression)...")
    backtest_logreg = run_backtest(model_type='logreg', save=True)
    
    # 2. Run backtest with gradient boosting
    print("\n[2/4] Running historical backtest (Gradient Boosting)...")
    backtest_gbm = run_backtest(model_type='gbm', save=True)
    
    # 3. Generate predictions for specified year
    predict_year_val = year or MAX_YEAR
    print(f"\n[3/4] Generating predictions for {predict_year_val}...")
    predictions = predict_year(predict_year_val, model_type='logreg')
    
    # 4. Run Monte Carlo simulation
    print(f"\n[4/4] Running Monte Carlo simulation for {predict_year_val}...")
    simulation = run_simulation(predict_year_val, n_simulations=5000)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\nBacktest Results (Logistic Regression):")
    print(f"  Mean Champion Rank: {backtest_logreg['mean_champion_rank']:.1f}")
    print(f"  Top-5 Rate: {backtest_logreg['top_5_rate']*100:.1f}%")
    print(f"  Top-10 Rate: {backtest_logreg['top_10_rate']*100:.1f}%")
    
    print(f"\nBacktest Results (Gradient Boosting):")
    print(f"  Mean Champion Rank: {backtest_gbm['mean_champion_rank']:.1f}")
    print(f"  Top-5 Rate: {backtest_gbm['top_5_rate']*100:.1f}%")
    print(f"  Top-10 Rate: {backtest_gbm['top_10_rate']*100:.1f}%")
    
    print(f"\nTop 5 Predictions for {predict_year_val}:")
    for i, pred in enumerate(predictions.get('top_5', [])[:5], 1):
        print(f"  {i}. {pred['TEAM']} (Seed {int(pred['SEED'])}): {pred['PROB']:.1%}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'backtest_logreg': backtest_logreg,
        'backtest_gbm': backtest_gbm,
        'predictions': predictions,
        'simulation': simulation
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='NCAA Championship Prediction Pipeline'
    )
    
    parser.add_argument(
        '--backtest', 
        action='store_true',
        help='Run historical backtest only'
    )
    
    parser.add_argument(
        '--predict',
        type=int,
        metavar='YEAR',
        help='Generate predictions for specific year'
    )
    
    parser.add_argument(
        '--simulate',
        type=int,
        metavar='YEAR', 
        help='Run Monte Carlo simulation for year'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='logreg',
        choices=['logreg', 'gbm'],
        help='Model type (default: logreg)'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations (default: 10000)'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate function
    if args.backtest:
        run_backtest(model_type=args.model)
    elif args.predict:
        predict_year(args.predict, model_type=args.model)
    elif args.simulate:
        run_simulation(args.simulate, n_simulations=args.simulations)
    else:
        # Run full pipeline
        run_full_pipeline(year=args.predict)


if __name__ == "__main__":
    main()
