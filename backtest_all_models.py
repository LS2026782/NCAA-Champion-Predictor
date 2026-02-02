"""
Full backtest comparing ALL model variants:
1. Original (baseline)
2. Enhanced (pattern-based features)
3. Momentum (late-season + conf tourney features)
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.features.enhanced_builder import EnhancedFeatureBuilder
from src.features.momentum_builder import MomentumFeatureBuilder
from src.models.champion_model import ChampionPredictor
from config.settings import FIRST_TEST_YEAR


def backtest_model(builder_class, model_name, loader, full_data):
    """Run backtest with given feature builder."""
    
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        builder = builder_class()
        
        # Handle different builder signatures
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
            'seed': int(champ['SEED']),
            'rank': int(champ['RANK']),
            'prob': champ['PROB'],
            'top_1': test_feat.nlargest(1, 'PROB').iloc[0]['TEAM']
        })
    
    return results


def main():
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON BACKTEST")
    print("="*80)
    
    # Load data once
    print("\nLoading data...")
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    # Run all backtests
    models = {
        'Original': FeatureBuilder,
        'Enhanced': EnhancedFeatureBuilder,
        'Momentum': MomentumFeatureBuilder
    }
    
    all_results = {}
    
    for name, builder_class in models.items():
        print(f"\n[{name}] Running backtest...")
        all_results[name] = backtest_model(builder_class, name, loader, full_data)
    
    # Year-by-year comparison
    print("\n" + "="*80)
    print("YEAR-BY-YEAR CHAMPION RANK COMPARISON")
    print("="*80)
    
    years = [r['year'] for r in all_results['Original']]
    
    print(f"\n{'Year':<6} {'Champion':<20} {'Orig':>8} {'Enh':>8} {'Mom':>8} {'Best':>10}")
    print("-"*70)
    
    for i, year in enumerate(years):
        champ = all_results['Original'][i]['champion']
        orig = all_results['Original'][i]['rank']
        enh = all_results['Enhanced'][i]['rank']
        mom = all_results['Momentum'][i]['rank']
        
        best_rank = min(orig, enh, mom)
        if best_rank == orig and best_rank == enh and best_rank == mom:
            best = "TIE"
        elif best_rank == orig:
            best = "ORIG" if orig < enh and orig < mom else "TIE"
        elif best_rank == enh:
            best = "ENH" if enh < orig and enh < mom else "TIE"
        else:
            best = "MOM" if mom < orig and mom < enh else "TIE"
            
        print(f"{year:<6} {champ:<20} {orig:>8} {enh:>8} {mom:>8} {best:>10}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Original':>12} {'Enhanced':>12} {'Momentum':>12}")
    print("-"*65)
    
    # Extract ranks separately to avoid modifying dict during iteration
    orig_ranks = [r['rank'] for r in all_results['Original']]
    enh_ranks = [r['rank'] for r in all_results['Enhanced']]
    mom_ranks = [r['rank'] for r in all_results['Momentum']]
    
    metrics = [
        ('Mean Rank', lambda r: np.mean(r)),
        ('Median Rank', lambda r: np.median(r)),
        ('Std Dev', lambda r: np.std(r)),
        ('Best Rank', lambda r: min(r)),
        ('Worst Rank', lambda r: max(r)),
    ]
    
    for metric_name, func in metrics:
        print(f"{metric_name:<25} {func(orig_ranks):>12.2f} {func(enh_ranks):>12.2f} {func(mom_ranks):>12.2f}")
    
    # Top-K rates
    print()
    for k in [1, 5, 10]:
        orig_pct = sum(r <= k for r in orig_ranks) / len(orig_ranks) * 100
        enh_pct = sum(r <= k for r in enh_ranks) / len(enh_ranks) * 100
        mom_pct = sum(r <= k for r in mom_ranks) / len(mom_ranks) * 100
        print(f"{'Top-' + str(k) + ' Rate':<25} {orig_pct:>11.1f}% {enh_pct:>11.1f}% {mom_pct:>11.1f}%")
    
    # Win counts
    print("\n" + "="*80)
    print("HEAD-TO-HEAD WINS (which model ranked champion highest)")
    print("="*80)
    
    orig_wins = 0
    enh_wins = 0
    mom_wins = 0
    ties = 0
    
    for i in range(len(years)):
        orig = all_results['Original'][i]['rank']
        enh = all_results['Enhanced'][i]['rank']
        mom = all_results['Momentum'][i]['rank']
        
        best = min(orig, enh, mom)
        winners = []
        if orig == best: winners.append('orig')
        if enh == best: winners.append('enh')
        if mom == best: winners.append('mom')
        
        if len(winners) == 1:
            if 'orig' in winners: orig_wins += 1
            if 'enh' in winners: enh_wins += 1
            if 'mom' in winners: mom_wins += 1
        else:
            ties += 1
    
    print(f"\nOriginal wins: {orig_wins}")
    print(f"Enhanced wins: {enh_wins}")
    print(f"Momentum wins: {mom_wins}")
    print(f"Ties: {ties}")
    
    # Best model recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    means = {
        'Original': np.mean(orig_ranks),
        'Enhanced': np.mean(enh_ranks),
        'Momentum': np.mean(mom_ranks)
    }
    
    medians = {
        'Original': np.median(orig_ranks),
        'Enhanced': np.median(enh_ranks),
        'Momentum': np.median(mom_ranks)
    }
    
    best_mean = min(means, key=means.get)
    best_median = min(medians, key=medians.get)
    
    print(f"\nBest by Mean Rank: {best_mean} ({means[best_mean]:.2f})")
    print(f"Best by Median Rank: {best_median} ({medians[best_median]:.1f})")
    
    if best_mean == best_median:
        print(f"\n*** RECOMMENDED MODEL: {best_mean} ***")
    else:
        print(f"\n*** Close call between {best_mean} (mean) and {best_median} (median) ***")


if __name__ == "__main__":
    main()
