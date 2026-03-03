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
from typing import List, Optional, TypedDict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    MIN_YEAR, MAX_YEAR, FIRST_TEST_YEAR,
    RESULTS_DIR, MODEL_FEATURES
)
from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.champion_model import (
    ChampionPredictor, EnsembleChampionPredictor, compute_era_weights
)
from src.evaluation.backtester import Backtester
from src.simulation.monte_carlo import BracketSimulator


# ---------------------------------------------------------------------------
# Return-type contracts
# ---------------------------------------------------------------------------

class BacktestResult(TypedDict):
    model_type: str
    n_years: int
    years_tested: List[int]
    mean_champion_rank: float
    median_champion_rank: float
    std_champion_rank: float
    min_champion_rank: int
    max_champion_rank: int
    top_1_rate: float
    top_5_rate: float
    top_10_rate: float
    top_25_rate: float
    mean_champion_prob: float
    min_champion_prob: float
    max_champion_prob: float
    mean_brier_score: float
    mean_log_loss: float
    year_details: List[dict]


class PredictionResult(TypedDict):
    year: int
    model_type: str
    top_5: List[dict]
    champion_rank: Optional[int]


class SimulationResult(TypedDict):
    year: int
    n_simulations: int
    top_5: list


class PipelineResult(TypedDict):
    backtest_logreg: BacktestResult
    backtest_gbm: BacktestResult
    predictions: PredictionResult
    simulation: SimulationResult


def run_backtest(
    model_type: str = 'logreg',
    save: bool = True,
    results_dir: Optional[Path] = None,
) -> BacktestResult:
    """
    Run full historical backtest.

    Args:
        model_type: 'logreg' or 'gbm'
        save: Whether to save results to file
        results_dir: Directory for output files (default: RESULTS_DIR from config)

    Returns:
        BacktestResult summary dict
    """
    output_dir = results_dir if results_dir is not None else RESULTS_DIR

    print("\n" + "="*70)
    print("RUNNING HISTORICAL BACKTEST")
    print("="*70)

    backtester = Backtester(model_type=model_type)
    backtester.run_backtest()
    backtester.print_report()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"backtest_{model_type}_{datetime.now().strftime('%Y%m%d')}.json"
        backtester.save_results(str(filepath))

    return backtester.get_summary()


def predict_year(
    year: int,
    model_type: str = 'logreg',
    results_dir: Optional[Path] = None,
) -> PredictionResult:
    """
    Generate predictions for a specific year.

    Args:
        year: Tournament year to predict
        model_type: Model type to use
        results_dir: Directory for output files (default: RESULTS_DIR from config)

    Returns:
        PredictionResult dict

    Raises:
        ValueError: If year is not present in the loaded dataset
    """
    output_dir = results_dir if results_dir is not None else RESULTS_DIR

    print("\n" + "="*70)
    print(f"GENERATING PREDICTIONS FOR {year}")
    print("="*70)

    # Load data
    loader = DataLoader()
    loader.load_all()

    # Fail fast — a silent empty-dict return masks configuration errors and
    # makes the calling code believe the pipeline succeeded.
    available_years = loader.get_years()
    if year not in available_years:
        raise ValueError(
            f"Year {year} not available. Choose from: {sorted(available_years)}"
        )
    
    # Get train/test split
    train_df, test_df = loader.get_train_test_split(year)
    
    # Build features
    builder = FeatureBuilder()
    train_df_feat, X_train = builder.build_features(train_df, fit_scaler=True)
    test_df_feat, X_test = builder.build_features(test_df, fit_scaler=False)
    
    y_train = train_df_feat['IS_CHAMPION'].values
    
    # Build season groups and era weights for temporal CV and concept drift
    season_groups = train_df_feat['YEAR'].values if 'YEAR' in train_df_feat.columns else None
    era_weights = compute_era_weights(season_groups) if season_groups is not None else None
    
    # Train model
    print(f"\nTraining {model_type.upper()} model...")
    model = ChampionPredictor(model_type=model_type)
    model.fit(X_train, y_train, feature_names=builder.get_feature_names(),
              season_groups=season_groups, era_weights=era_weights)
    
    # Predict — normalized so field sums to 1.0 (one winner per tournament)
    probs = model.predict_proba_normalized(X_test)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / f"predictions_{year}_{model_type}.csv"
    predictions[['RANK', 'TEAM', 'SEED', 'PROB', 'IS_CHAMPION']].to_csv(
        predictions_file, index=False
    )
    print(f"\nPredictions saved to {predictions_file}")

    champ_mask = predictions['IS_CHAMPION'] == 1
    champion_rank: Optional[int] = (
        int(predictions.loc[champ_mask, 'RANK'].values[0])
        if champ_mask.any() else None
    )

    return PredictionResult(
        year=year,
        model_type=model_type,
        top_5=predictions.head(5)[['TEAM', 'SEED', 'PROB']].to_dict('records'),
        champion_rank=champion_rank,
    )


def run_simulation(
    year: int,
    n_simulations: int = 10000,
    results_dir: Optional[Path] = None,
) -> SimulationResult:
    """
    Run Monte Carlo bracket simulation for a year.

    Args:
        year: Tournament year
        n_simulations: Number of simulations
        results_dir: Directory for output files (default: RESULTS_DIR from config)

    Returns:
        SimulationResult dict
    """
    output_dir = results_dir if results_dir is not None else RESULTS_DIR

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
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_file = output_dir / f"simulation_{year}.json"
    with open(sim_file, 'w') as f:
        json.dump({
            'year': year,
            'n_simulations': n_simulations,
            'championship_odds': {k: float(v) for k, v in sorted_probs}
        }, f, indent=2)
    print(f"\nSimulation results saved to {sim_file}")

    return SimulationResult(
        year=year,
        n_simulations=n_simulations,
        top_5=sorted_probs[:5],
    )


def run_full_pipeline(
    year: Optional[int] = None,
    results_dir: Optional[Path] = None,
) -> PipelineResult:
    """
    Run the complete prediction pipeline.

    Args:
        year: Year to predict (default: most recent year in dataset)
        results_dir: Directory for all output files (default: RESULTS_DIR from config)

    Returns:
        PipelineResult with backtest summaries, predictions, and simulation
    """
    output_dir = results_dir if results_dir is not None else RESULTS_DIR

    print("\n" + "="*70)
    print("NCAA CHAMPIONSHIP PREDICTION PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Run backtest with logistic regression
    print("\n[1/4] Running historical backtest (Logistic Regression)...")
    backtest_logreg = run_backtest(model_type='logreg', save=True, results_dir=output_dir)

    # 2. Run backtest with gradient boosting
    print("\n[2/4] Running historical backtest (Gradient Boosting)...")
    backtest_gbm = run_backtest(model_type='gbm', save=True, results_dir=output_dir)

    # 3. Generate predictions for specified year
    predict_year_val = year or MAX_YEAR
    print(f"\n[3/4] Generating predictions for {predict_year_val}...")
    predictions = predict_year(predict_year_val, model_type='logreg', results_dir=output_dir)

    # 4. Run Monte Carlo simulation
    print(f"\n[4/4] Running Monte Carlo simulation for {predict_year_val}...")
    simulation = run_simulation(predict_year_val, n_simulations=5000, results_dir=output_dir)

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

    return PipelineResult(
        backtest_logreg=backtest_logreg,
        backtest_gbm=backtest_gbm,
        predictions=predictions,
        simulation=simulation,
    )


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
