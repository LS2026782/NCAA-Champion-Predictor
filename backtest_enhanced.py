"""
Backtest the enhanced model vs original model.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.features.enhanced_builder import EnhancedFeatureBuilder
from src.models.champion_model import ChampionPredictor
from config.settings import FIRST_TEST_YEAR, MAX_YEAR

def backtest_model(builder_class, model_name):
    """Run backtest with given feature builder."""
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        builder = builder_class()
        
        if hasattr(builder, 'build_features'):
            if 'all_data' in builder.build_features.__code__.co_varnames:
                train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
                test_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
            else:
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
    print("="*70)
    print("BACKTEST COMPARISON: ORIGINAL vs ENHANCED MODEL")
    print("="*70)
    
    # Original model
    print("\n[1/2] Running ORIGINAL model backtest...")
    original_results = backtest_model(FeatureBuilder, "Original")
    
    # Enhanced model
    print("\n[2/2] Running ENHANCED model backtest...")
    enhanced_results = backtest_model(EnhancedFeatureBuilder, "Enhanced")
    
    # Compare results
    print("\n" + "="*70)
    print("YEAR-BY-YEAR COMPARISON")
    print("="*70)
    print(f"{'Year':<6} {'Champion':<20} {'Orig Rank':>10} {'Enh Rank':>10} {'Better?':>10}")
    print("-"*60)
    
    orig_better = 0
    enh_better = 0
    same = 0
    
    for orig, enh in zip(original_results, enhanced_results):
        if orig['rank'] < enh['rank']:
            better = "ORIG"
            orig_better += 1
        elif enh['rank'] < orig['rank']:
            better = "ENH"
            enh_better += 1
        else:
            better = "TIE"
            same += 1
            
        print(f"{orig['year']:<6} {orig['champion']:<20} {orig['rank']:>10} {enh['rank']:>10} {better:>10}")
    
    # Summary stats
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    orig_ranks = [r['rank'] for r in original_results]
    enh_ranks = [r['rank'] for r in enhanced_results]
    
    print(f"\n{'Metric':<30} {'Original':>15} {'Enhanced':>15}")
    print("-"*60)
    print(f"{'Mean Champion Rank':<30} {np.mean(orig_ranks):>15.2f} {np.mean(enh_ranks):>15.2f}")
    print(f"{'Median Champion Rank':<30} {np.median(orig_ranks):>15.1f} {np.median(enh_ranks):>15.1f}")
    print(f"{'Top-1 Rate':<30} {sum(r['rank']==1 for r in original_results)/len(original_results)*100:>14.1f}% {sum(r['rank']==1 for r in enhanced_results)/len(enhanced_results)*100:>14.1f}%")
    print(f"{'Top-5 Rate':<30} {sum(r['rank']<=5 for r in original_results)/len(original_results)*100:>14.1f}% {sum(r['rank']<=5 for r in enhanced_results)/len(enhanced_results)*100:>14.1f}%")
    print(f"{'Top-10 Rate':<30} {sum(r['rank']<=10 for r in original_results)/len(original_results)*100:>14.1f}% {sum(r['rank']<=10 for r in enhanced_results)/len(enhanced_results)*100:>14.1f}%")
    
    print(f"\nHead-to-head: Original won {orig_better}, Enhanced won {enh_better}, Ties {same}")
    
    # Which years did enhanced improve?
    print("\n" + "="*70)
    print("YEARS WHERE ENHANCED MODEL IMPROVED")
    print("="*70)
    
    for orig, enh in zip(original_results, enhanced_results):
        if enh['rank'] < orig['rank']:
            improvement = orig['rank'] - enh['rank']
            print(f"  {orig['year']} {orig['champion']:<20}: {orig['rank']} -> {enh['rank']} (+{improvement} positions)")
    
    print("\n" + "="*70)
    print("YEARS WHERE ENHANCED MODEL GOT WORSE")
    print("="*70)
    
    for orig, enh in zip(original_results, enhanced_results):
        if enh['rank'] > orig['rank']:
            decline = enh['rank'] - orig['rank']
            print(f"  {orig['year']} {orig['champion']:<20}: {orig['rank']} -> {enh['rank']} (-{decline} positions)")


if __name__ == "__main__":
    main()
