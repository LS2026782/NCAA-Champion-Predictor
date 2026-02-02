"""
Test both Ensemble and Gradient Boosting approaches.

1. Ensemble: Combine predictions from Original, Ultimate, and Final models
2. Gradient Boosting: Test HistGradientBoostingClassifier with various settings
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.features.ultimate_builder import UltimateFeatureBuilder
from src.features.final_builder import FinalFeatureBuilder
from src.models.champion_model import ChampionPredictor
from config.settings import FIRST_TEST_YEAR

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


def run_single_model(builder_class, train_df, test_df, full_data):
    """Run a single model and return predictions."""
    builder = builder_class()
    try:
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
        test_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
    except TypeError:
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True)
        test_feat, X_test = builder.build_features(test_df, fit_scaler=False)
    
    y_train = train_feat['IS_CHAMPION'].values
    
    model = ChampionPredictor(model_type='logreg')
    model.fit(X_train, y_train, feature_names=builder.get_feature_names())
    
    probs = model.predict_proba(X_test)
    return test_feat, probs


def run_gradient_boosting(builder_class, train_df, test_df, full_data, max_depth=3, learning_rate=0.05):
    """Run gradient boosting model."""
    builder = builder_class()
    try:
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
        test_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
    except TypeError:
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True)
        test_feat, X_test = builder.build_features(test_df, fit_scaler=False)
    
    y_train = train_feat['IS_CHAMPION'].values
    
    # Gradient Boosting with regularization
    gb = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=100,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=42,
        class_weight='balanced'
    )
    
    # Calibrate for better probabilities
    calibrated = CalibratedClassifierCV(gb, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)
    
    probs = calibrated.predict_proba(X_test)[:, 1]
    return test_feat, probs


def backtest_ensemble():
    """Backtest ensemble of all models."""
    print("="*70)
    print("PART 1: ENSEMBLE MODEL BACKTEST")
    print("="*70)
    print("Combining: Original + Ultimate + Final (weighted average)")
    print()
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    
    # Different weighting schemes to test
    weight_schemes = {
        'Equal': [1/3, 1/3, 1/3],
        'Ultimate-Heavy': [0.2, 0.5, 0.3],
        'Final-Heavy': [0.2, 0.3, 0.5],
        'Original-Heavy': [0.5, 0.25, 0.25],
    }
    
    all_results = {name: [] for name in weight_schemes}
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        # Get predictions from each model
        test_feat_orig, probs_orig = run_single_model(FeatureBuilder, train_df, test_df, full_data)
        test_feat_ult, probs_ult = run_single_model(UltimateFeatureBuilder, train_df, test_df, full_data)
        test_feat_final, probs_final = run_single_model(FinalFeatureBuilder, train_df, test_df, full_data)
        
        # Try different weight schemes
        for scheme_name, weights in weight_schemes.items():
            ensemble_probs = weights[0] * probs_orig + weights[1] * probs_ult + weights[2] * probs_final
            
            test_feat = test_feat_orig.copy()
            test_feat['PROB'] = ensemble_probs
            test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
            
            champ = test_feat[test_feat['IS_CHAMPION'] == 1].iloc[0]
            all_results[scheme_name].append({
                'year': test_year,
                'champion': champ['TEAM'],
                'rank': int(champ['RANK'])
            })
    
    # Print results for each scheme
    print(f"{'Year':<6} {'Champion':<18} ", end='')
    for name in weight_schemes:
        print(f"{name[:8]:>10}", end='')
    print()
    print("-"*70)
    
    for i, year in enumerate(years):
        champ = all_results['Equal'][i]['champion']
        print(f"{year:<6} {champ:<18} ", end='')
        for name in weight_schemes:
            rank = all_results[name][i]['rank']
            print(f"{rank:>10}", end='')
        print()
    
    # Summary
    print("\n" + "="*70)
    print("ENSEMBLE SUMMARY")
    print("="*70)
    print(f"\n{'Scheme':<20} {'Mean':>8} {'Median':>8} {'Top-5':>10} {'Top-10':>10}")
    print("-"*60)
    
    for name, results in all_results.items():
        ranks = [r['rank'] for r in results]
        mean = np.mean(ranks)
        median = np.median(ranks)
        top5 = sum(r <= 5 for r in ranks) / len(ranks) * 100
        top10 = sum(r <= 10 for r in ranks) / len(ranks) * 100
        print(f"{name:<20} {mean:>8.2f} {median:>8.1f} {top5:>9.1f}% {top10:>9.1f}%")
    
    return all_results


def backtest_gradient_boosting():
    """Backtest gradient boosting with various settings."""
    print("\n" + "="*70)
    print("PART 2: GRADIENT BOOSTING BACKTEST")
    print("="*70)
    print("Testing HistGradientBoostingClassifier with Final features")
    print()
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    
    # Different GB configurations
    configs = {
        'GB-Shallow': {'max_depth': 2, 'learning_rate': 0.05},
        'GB-Medium': {'max_depth': 3, 'learning_rate': 0.05},
        'GB-Deep': {'max_depth': 4, 'learning_rate': 0.03},
        'GB-Conservative': {'max_depth': 2, 'learning_rate': 0.01},
    }
    
    all_results = {name: [] for name in configs}
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        for config_name, params in configs.items():
            try:
                test_feat, probs = run_gradient_boosting(
                    FinalFeatureBuilder, train_df, test_df, full_data,
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate']
                )
                
                test_feat = test_feat.copy()
                test_feat['PROB'] = probs
                test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
                
                champ = test_feat[test_feat['IS_CHAMPION'] == 1].iloc[0]
                all_results[config_name].append({
                    'year': test_year,
                    'champion': champ['TEAM'],
                    'rank': int(champ['RANK'])
                })
            except Exception as e:
                print(f"  {test_year} {config_name}: Error - {e}")
                all_results[config_name].append({
                    'year': test_year,
                    'champion': 'ERROR',
                    'rank': 68
                })
    
    # Print results
    print(f"{'Year':<6} {'Champion':<18} ", end='')
    for name in configs:
        print(f"{name[:10]:>12}", end='')
    print()
    print("-"*75)
    
    for i, year in enumerate(years):
        champ = all_results['GB-Shallow'][i]['champion']
        print(f"{year:<6} {champ:<18} ", end='')
        for name in configs:
            rank = all_results[name][i]['rank']
            print(f"{rank:>12}", end='')
        print()
    
    # Summary
    print("\n" + "="*70)
    print("GRADIENT BOOSTING SUMMARY")
    print("="*70)
    print(f"\n{'Config':<20} {'Mean':>8} {'Median':>8} {'Top-5':>10} {'Top-10':>10}")
    print("-"*60)
    
    for name, results in all_results.items():
        ranks = [r['rank'] for r in results]
        mean = np.mean(ranks)
        median = np.median(ranks)
        top5 = sum(r <= 5 for r in ranks) / len(ranks) * 100
        top10 = sum(r <= 10 for r in ranks) / len(ranks) * 100
        print(f"{name:<20} {mean:>8.2f} {median:>8.1f} {top5:>9.1f}% {top10:>9.1f}%")
    
    return all_results


def test_2025():
    """Test best approaches on 2025."""
    print("\n" + "="*70)
    print("PART 3: 2025 PREDICTION TEST")
    print("="*70)
    
    df = pd.read_csv('data/raw/KenPom Barttorvik.csv')
    teams_2025 = df[(df['YEAR'] == 2025) & (df['SEED'].notna())].copy()
    train_df = df[(df['YEAR'] < 2025) & (df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()
    train_df['SEED_NUM'] = train_df['SEED'].astype(float)
    teams_2025['SEED_NUM'] = teams_2025['SEED'].astype(float)
    train_df['IS_CHAMPION'] = (train_df['ROUND'] == 1).astype(int)
    
    full_data = df[(df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()
    full_data['SEED_NUM'] = full_data['SEED'].astype(float)
    
    results = {}
    
    # Individual models
    for name, builder_class in [('Original', FeatureBuilder), ('Ultimate', UltimateFeatureBuilder), ('Final', FinalFeatureBuilder)]:
        test_feat, probs = run_single_model(builder_class, train_df, teams_2025, full_data)
        test_feat = test_feat.copy()
        test_feat['PROB'] = probs
        test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
        results[name] = test_feat
    
    # Ensemble (Equal weights)
    probs_ensemble = (results['Original']['PROB'].values + results['Ultimate']['PROB'].values + results['Final']['PROB'].values) / 3
    results['Ensemble'] = results['Original'].copy()
    results['Ensemble']['PROB'] = probs_ensemble
    results['Ensemble']['RANK'] = results['Ensemble']['PROB'].rank(ascending=False).astype(int)
    
    # Gradient Boosting
    test_feat, probs_gb = run_gradient_boosting(FinalFeatureBuilder, train_df, teams_2025, full_data, max_depth=2, learning_rate=0.05)
    results['GB-Shallow'] = test_feat.copy()
    results['GB-Shallow']['PROB'] = probs_gb
    results['GB-Shallow']['RANK'] = results['GB-Shallow']['PROB'].rank(ascending=False).astype(int)
    
    # Print comparison
    print(f"\n{'Model':<15} {'Florida':>10} {'Houston':>10} {'Auburn':>10} {'Duke':>10}")
    print("-"*60)
    
    for name in ['Original', 'Ultimate', 'Final', 'Ensemble', 'GB-Shallow']:
        florida = int(results[name][results[name]['TEAM'] == 'Florida']['RANK'].values[0])
        houston = int(results[name][results[name]['TEAM'] == 'Houston']['RANK'].values[0])
        auburn = int(results[name][results[name]['TEAM'] == 'Auburn']['RANK'].values[0])
        duke = int(results[name][results[name]['TEAM'] == 'Duke']['RANK'].values[0])
        print(f"{name:<15} {florida:>10} {houston:>10} {auburn:>10} {duke:>10}")
    
    print("\nFlorida = 2025 ACTUAL CHAMPION")
    
    # Show Ensemble top 10
    print("\n" + "="*70)
    print("ENSEMBLE MODEL - TOP 10 FOR 2025")
    print("="*70)
    
    top10 = results['Ensemble'].nsmallest(10, 'RANK')[['TEAM', 'SEED', 'PROB', 'RANK']]
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        marker = " <-- CHAMPION" if row['TEAM'] == 'Florida' else ""
        print(f"  {i:2d}. {row['TEAM']:<20} (Seed {int(row['SEED'])}) Prob: {row['PROB']:.3f}{marker}")


if __name__ == "__main__":
    ensemble_results = backtest_ensemble()
    gb_results = backtest_gradient_boosting()
    test_2025()
    
    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON - ALL APPROACHES")
    print("="*70)
    
    print(f"\n{'Approach':<25} {'Mean Rank':>12} {'Top-5 Rate':>12} {'Top-10 Rate':>12}")
    print("-"*65)
    
    # Best from each category
    comparisons = [
        ('Original (Baseline)', 7.91, 54.5, 72.7),
        ('Ultimate', 5.64, 63.6, 81.8),
        ('Final', 5.64, 63.6, 81.8),
    ]
    
    # Add ensemble results
    for name, results in ensemble_results.items():
        ranks = [r['rank'] for r in results]
        comparisons.append((f'Ensemble-{name}', np.mean(ranks), 
                           sum(r<=5 for r in ranks)/len(ranks)*100,
                           sum(r<=10 for r in ranks)/len(ranks)*100))
    
    # Add GB results  
    for name, results in gb_results.items():
        ranks = [r['rank'] for r in results]
        comparisons.append((name, np.mean(ranks),
                           sum(r<=5 for r in ranks)/len(ranks)*100,
                           sum(r<=10 for r in ranks)/len(ranks)*100))
    
    # Sort by mean rank
    comparisons.sort(key=lambda x: x[1])
    
    for name, mean, top5, top10 in comparisons:
        print(f"{name:<25} {mean:>12.2f} {top5:>11.1f}% {top10:>11.1f}%")
