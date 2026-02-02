"""
Final comprehensive comparison of all models including Ultimate.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.features.momentum_builder import MomentumFeatureBuilder
from src.features.ultimate_builder import UltimateFeatureBuilder
from src.models.champion_model import ChampionPredictor
from config.settings import FIRST_TEST_YEAR


def backtest_model(builder_class, loader, full_data):
    """Run backtest."""
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
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
        test_feat = test_feat.copy()
        test_feat['PROB'] = probs
        test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
        
        champ = test_feat[test_feat['IS_CHAMPION'] == 1].iloc[0]
        results.append({
            'year': test_year,
            'champion': champ['TEAM'],
            'rank': int(champ['RANK'])
        })
    
    return results


def main():
    print("="*80)
    print("FINAL MODEL COMPARISON")
    print("="*80)
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    models = {
        'Original': FeatureBuilder,
        'Momentum': MomentumFeatureBuilder,
        'Ultimate': UltimateFeatureBuilder
    }
    
    all_results = {}
    for name, builder_class in models.items():
        print(f"\nRunning {name}...")
        all_results[name] = backtest_model(builder_class, loader, full_data)
    
    # Year-by-year
    print("\n" + "="*80)
    print("YEAR-BY-YEAR COMPARISON")
    print("="*80)
    
    print(f"\n{'Year':<6} {'Champion':<20} {'Orig':>8} {'Mom':>8} {'Ult':>8} {'Best':>10}")
    print("-"*65)
    
    for i, year in enumerate([r['year'] for r in all_results['Original']]):
        champ = all_results['Original'][i]['champion']
        orig = all_results['Original'][i]['rank']
        mom = all_results['Momentum'][i]['rank']
        ult = all_results['Ultimate'][i]['rank']
        
        best = min(orig, mom, ult)
        winner = []
        if orig == best: winner.append('O')
        if mom == best: winner.append('M')
        if ult == best: winner.append('U')
        
        print(f"{year:<6} {champ:<20} {orig:>8} {mom:>8} {ult:>8} {'/'.join(winner):>10}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Original':>12} {'Momentum':>12} {'Ultimate':>12}")
    print("-"*65)
    
    for name in models:
        ranks = [r['rank'] for r in all_results[name]]
        all_results[name + '_ranks'] = ranks
    
    metrics = {
        'Mean Rank': lambda r: np.mean(r),
        'Median Rank': lambda r: np.median(r),
        'Std Dev': lambda r: np.std(r),
        'Best (Min)': lambda r: min(r),
        'Worst (Max)': lambda r: max(r),
    }
    
    for metric, func in metrics.items():
        orig = func(all_results['Original_ranks'])
        mom = func(all_results['Momentum_ranks'])
        ult = func(all_results['Ultimate_ranks'])
        print(f"{metric:<25} {orig:>12.2f} {mom:>12.2f} {ult:>12.2f}")
    
    print()
    for k in [1, 3, 5, 10]:
        orig = sum(r <= k for r in all_results['Original_ranks']) / len(all_results['Original_ranks']) * 100
        mom = sum(r <= k for r in all_results['Momentum_ranks']) / len(all_results['Momentum_ranks']) * 100
        ult = sum(r <= k for r in all_results['Ultimate_ranks']) / len(all_results['Ultimate_ranks']) * 100
        print(f"{'Top-' + str(k) + ' Rate':<25} {orig:>11.1f}% {mom:>11.1f}% {ult:>11.1f}%")
    
    # Best model
    print("\n" + "="*80)
    print("WINNER")
    print("="*80)
    
    means = {
        'Original': np.mean(all_results['Original_ranks']),
        'Momentum': np.mean(all_results['Momentum_ranks']),
        'Ultimate': np.mean(all_results['Ultimate_ranks'])
    }
    
    best = min(means, key=means.get)
    print(f"\n*** BEST MODEL BY MEAN RANK: {best} ({means[best]:.2f}) ***")
    
    # Test on 2025
    print("\n" + "="*80)
    print("2025 PREDICTION TEST (Florida = Actual Champion)")
    print("="*80)
    
    df = pd.read_csv('data/raw/KenPom Barttorvik.csv')
    teams_2025 = df[(df['YEAR'] == 2025) & (df['SEED'].notna())].copy()
    train_df = df[(df['YEAR'] < 2025) & (df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()
    train_df['SEED_NUM'] = train_df['SEED'].astype(float)
    teams_2025['SEED_NUM'] = teams_2025['SEED'].astype(float)
    train_df['IS_CHAMPION'] = (train_df['ROUND'] == 1).astype(int)
    
    full_data_2025 = df[(df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()
    full_data_2025['SEED_NUM'] = full_data_2025['SEED'].astype(float)
    
    print(f"\n{'Model':<15} {'Florida Rank':>15} {'Houston Rank':>15}")
    print("-"*50)
    
    for name, builder_class in models.items():
        builder = builder_class()
        try:
            train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data_2025)
            test_feat, X_test = builder.build_features(teams_2025, fit_scaler=False, all_data=full_data_2025)
        except TypeError:
            train_feat, X_train = builder.build_features(train_df, fit_scaler=True)
            test_feat, X_test = builder.build_features(teams_2025, fit_scaler=False)
        
        y_train = train_feat['IS_CHAMPION'].values
        model = ChampionPredictor(model_type='logreg')
        model.fit(X_train, y_train, feature_names=builder.get_feature_names())
        
        probs = model.predict_proba(X_test)
        test_feat = test_feat.copy()
        test_feat['PROB'] = probs
        test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
        
        florida = test_feat[test_feat['TEAM'] == 'Florida'].iloc[0]['RANK']
        houston = test_feat[test_feat['TEAM'] == 'Houston'].iloc[0]['RANK']
        
        print(f"{name:<15} {int(florida):>15} {int(houston):>15}")


if __name__ == "__main__":
    main()
