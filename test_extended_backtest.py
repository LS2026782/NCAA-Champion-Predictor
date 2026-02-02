"""
Test if backtesting further back in time helps.

Comparison:
1. Current: 2013-2024 (11 years)
2. Extended: 2010-2024 (14 years, but less aux data early)
3. Full: 2009-2024 (15 years, minimal training for early years)
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


def backtest_from_year(builder_class, start_year, name=""):
    """Run backtest starting from a specific year."""
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    # Get years for testing
    all_years = sorted([y for y in loader.get_years() if y >= start_year])
    
    results = []
    
    for test_year in all_years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        # Skip if not enough training data
        if len(train_df) < 100:
            continue
            
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
            'seed': int(champ['SEED']),
            'rank': int(champ['RANK']),
            'train_size': len(train_df)
        })
    
    return results


def main():
    print("="*80)
    print("EXTENDED BACKTEST ANALYSIS")
    print("="*80)
    print("Testing if going further back in time helps predictions")
    print()
    
    # Test different start years with Original model (most robust)
    start_years = [2009, 2010, 2011, 2012, 2013]
    
    all_results = {}
    
    print("Running backtests...")
    for start in start_years:
        results = backtest_from_year(FeatureBuilder, start, f"From {start}")
        all_results[start] = results
        print(f"  {start}-2024: {len(results)} test years")
    
    # Detailed year-by-year comparison
    print("\n" + "="*80)
    print("YEAR-BY-YEAR CHAMPION RANKS (Original Model)")
    print("="*80)
    
    # Get all unique years
    all_test_years = sorted(set(r['year'] for results in all_results.values() for r in results))
    
    print(f"\n{'Year':<6} {'Champion':<20} {'Seed':>5} ", end='')
    for start in start_years:
        print(f"{'From'+str(start):>10}", end='')
    print(f" {'Train N':>10}")
    print("-"*90)
    
    for year in all_test_years:
        # Get champion name from any result
        champ_info = None
        for start in start_years:
            for r in all_results[start]:
                if r['year'] == year:
                    champ_info = r
                    break
            if champ_info:
                break
        
        if champ_info:
            print(f"{year:<6} {champ_info['champion']:<20} {champ_info['seed']:>5} ", end='')
            
            for start in start_years:
                rank = '-'
                for r in all_results[start]:
                    if r['year'] == year:
                        rank = r['rank']
                        train_n = r['train_size']
                        break
                print(f"{str(rank):>10}", end='')
            
            print(f" {train_n:>10}")
    
    # Summary stats for each start year
    print("\n" + "="*80)
    print("SUMMARY BY START YEAR")
    print("="*80)
    
    print(f"\n{'Start':<8} {'Years':>8} {'Mean':>8} {'Median':>8} {'Top-5':>10} {'Top-10':>10} {'Worst':>8}")
    print("-"*70)
    
    for start in start_years:
        results = all_results[start]
        ranks = [r['rank'] for r in results]
        
        if ranks:
            mean = np.mean(ranks)
            median = np.median(ranks)
            top5 = sum(r <= 5 for r in ranks) / len(ranks) * 100
            top10 = sum(r <= 10 for r in ranks) / len(ranks) * 100
            worst = max(ranks)
            
            print(f"{start:<8} {len(ranks):>8} {mean:>8.2f} {median:>8.1f} {top5:>9.1f}% {top10:>9.1f}% {worst:>8}")
    
    # Analysis of early vs late years
    print("\n" + "="*80)
    print("ERA ANALYSIS - Are older years harder to predict?")
    print("="*80)
    
    # Use the most complete backtest (from 2009)
    results_2009 = all_results[2009]
    
    early_years = [r for r in results_2009 if r['year'] <= 2015]
    late_years = [r for r in results_2009 if r['year'] > 2015]
    
    early_ranks = [r['rank'] for r in early_years]
    late_ranks = [r['rank'] for r in late_years]
    
    print(f"\nEarly Era (2009-2015): {len(early_ranks)} years")
    print(f"  Mean Rank:  {np.mean(early_ranks):.2f}")
    print(f"  Median:     {np.median(early_ranks):.1f}")
    print(f"  Top-5 Rate: {sum(r<=5 for r in early_ranks)/len(early_ranks)*100:.1f}%")
    
    print(f"\nLate Era (2016-2024): {len(late_ranks)} years")
    print(f"  Mean Rank:  {np.mean(late_ranks):.2f}")
    print(f"  Median:     {np.median(late_ranks):.1f}")
    print(f"  Top-5 Rate: {sum(r<=5 for r in late_ranks)/len(late_ranks)*100:.1f}%")
    
    # Champions by seed era comparison
    print("\n" + "="*80)
    print("CHAMPION SEEDS BY ERA")
    print("="*80)
    
    early_seeds = [r['seed'] for r in early_years]
    late_seeds = [r['seed'] for r in late_years]
    
    print(f"\nEarly Era (2009-2015):")
    print(f"  Average seed: {np.mean(early_seeds):.1f}")
    print(f"  Seeds: {early_seeds}")
    
    print(f"\nLate Era (2016-2024):")
    print(f"  Average seed: {np.mean(late_seeds):.1f}")
    print(f"  Seeds: {late_seeds}")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Find optimal start year
    best_start = min(start_years, key=lambda s: np.mean([r['rank'] for r in all_results[s]]))
    
    print(f"""
Analysis shows:
- More test years = more robust evaluation
- Earlier years may have different patterns
- Training data size affects early year predictions

Best start year by mean rank: {best_start}
    """)


if __name__ == "__main__":
    main()
