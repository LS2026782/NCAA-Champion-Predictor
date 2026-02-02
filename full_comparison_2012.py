"""
Full comparison of all approaches with 2012 start year.
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
    return test_feat, probs, builder.get_feature_names()


def run_gradient_boosting(builder_class, train_df, test_df, full_data):
    """Run gradient boosting model."""
    builder = builder_class()
    try:
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
        test_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
    except TypeError:
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True)
        test_feat, X_test = builder.build_features(test_df, fit_scaler=False)
    
    y_train = train_feat['IS_CHAMPION'].values
    
    gb = HistGradientBoostingClassifier(
        max_depth=2,
        learning_rate=0.05,
        max_iter=100,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=42,
        class_weight='balanced'
    )
    
    calibrated = CalibratedClassifierCV(gb, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)
    
    probs = calibrated.predict_proba(X_test)[:, 1]
    return test_feat, probs


def main():
    print("="*80)
    print(f"FULL MODEL COMPARISON (Starting from {FIRST_TEST_YEAR})")
    print("="*80)
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = sorted([y for y in loader.get_years() if y >= FIRST_TEST_YEAR])
    print(f"Test years: {years}")
    print(f"Total test years: {len(years)}")
    print()
    
    # Store all results
    all_results = {
        'Original': [],
        'Ultimate': [],
        'Final': [],
        'Ensemble-Equal': [],
        'Ensemble-Final-Heavy': [],
        'GB-Shallow': []
    }
    
    print("Running backtests...")
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        # Get predictions from base models
        test_feat_orig, probs_orig, _ = run_single_model(FeatureBuilder, train_df, test_df, full_data)
        test_feat_ult, probs_ult, _ = run_single_model(UltimateFeatureBuilder, train_df, test_df, full_data)
        test_feat_final, probs_final, _ = run_single_model(FinalFeatureBuilder, train_df, test_df, full_data)
        
        # Gradient Boosting
        try:
            _, probs_gb = run_gradient_boosting(FinalFeatureBuilder, train_df, test_df, full_data)
        except:
            probs_gb = probs_final  # Fallback
        
        # Ensemble predictions
        probs_equal = (probs_orig + probs_ult + probs_final) / 3
        probs_final_heavy = 0.2 * probs_orig + 0.3 * probs_ult + 0.5 * probs_final
        
        # Calculate ranks for each approach
        approaches = {
            'Original': probs_orig,
            'Ultimate': probs_ult,
            'Final': probs_final,
            'Ensemble-Equal': probs_equal,
            'Ensemble-Final-Heavy': probs_final_heavy,
            'GB-Shallow': probs_gb
        }
        
        for name, probs in approaches.items():
            test_feat = test_feat_orig.copy()
            test_feat['PROB'] = probs
            test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
            
            champ = test_feat[test_feat['IS_CHAMPION'] == 1].iloc[0]
            all_results[name].append({
                'year': test_year,
                'champion': champ['TEAM'],
                'seed': int(champ['SEED']),
                'rank': int(champ['RANK'])
            })
    
    # Year-by-year results
    print("\n" + "="*80)
    print("YEAR-BY-YEAR CHAMPION RANKS")
    print("="*80)
    
    header = f"{'Year':<6} {'Champion':<18} {'Seed':>5}"
    for name in all_results.keys():
        header += f" {name[:8]:>9}"
    print(header)
    print("-"*100)
    
    for i, year in enumerate(years):
        row = all_results['Original'][i]
        line = f"{year:<6} {row['champion']:<18} {row['seed']:>5}"
        for name in all_results.keys():
            rank = all_results[name][i]['rank']
            line += f" {rank:>9}"
        print(line)
    
    # Summary statistics
    print("\n" + "="*80)
    print("FINAL RANKINGS - ALL APPROACHES")
    print("="*80)
    
    summary = []
    for name, results in all_results.items():
        ranks = [r['rank'] for r in results]
        summary.append({
            'Approach': name,
            'Mean': np.mean(ranks),
            'Median': np.median(ranks),
            'Top-1': sum(r == 1 for r in ranks) / len(ranks) * 100,
            'Top-3': sum(r <= 3 for r in ranks) / len(ranks) * 100,
            'Top-5': sum(r <= 5 for r in ranks) / len(ranks) * 100,
            'Top-10': sum(r <= 10 for r in ranks) / len(ranks) * 100,
            'Worst': max(ranks)
        })
    
    # Sort by mean rank
    summary.sort(key=lambda x: x['Mean'])
    
    print(f"\n{'Rank':<5} {'Approach':<22} {'Mean':>8} {'Median':>8} {'Top-3':>8} {'Top-5':>8} {'Top-10':>9} {'Worst':>7}")
    print("-"*85)
    
    for i, s in enumerate(summary, 1):
        print(f"{i:<5} {s['Approach']:<22} {s['Mean']:>8.2f} {s['Median']:>8.1f} "
              f"{s['Top-3']:>7.1f}% {s['Top-5']:>7.1f}% {s['Top-10']:>8.1f}% {s['Worst']:>7}")
    
    # 2025 Test
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
    
    # Get 2025 predictions
    test_feat_orig, probs_orig, _ = run_single_model(FeatureBuilder, train_df, teams_2025, full_data_2025)
    test_feat_ult, probs_ult, _ = run_single_model(UltimateFeatureBuilder, train_df, teams_2025, full_data_2025)
    test_feat_final, probs_final, _ = run_single_model(FinalFeatureBuilder, train_df, teams_2025, full_data_2025)
    _, probs_gb = run_gradient_boosting(FinalFeatureBuilder, train_df, teams_2025, full_data_2025)
    
    probs_equal = (probs_orig + probs_ult + probs_final) / 3
    probs_final_heavy = 0.2 * probs_orig + 0.3 * probs_ult + 0.5 * probs_final
    
    approaches_2025 = {
        'Original': probs_orig,
        'Ultimate': probs_ult,
        'Final': probs_final,
        'Ensemble-Equal': probs_equal,
        'Ensemble-Final-Heavy': probs_final_heavy,
        'GB-Shallow': probs_gb
    }
    
    print(f"\n{'Model':<22} {'Florida':>10} {'Houston':>10} {'Auburn':>10} {'Duke':>10}")
    print("-"*65)
    
    for name, probs in approaches_2025.items():
        test_feat = test_feat_orig.copy()
        test_feat['PROB'] = probs
        test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
        
        florida = int(test_feat[test_feat['TEAM'] == 'Florida']['RANK'].values[0])
        houston = int(test_feat[test_feat['TEAM'] == 'Houston']['RANK'].values[0])
        auburn = int(test_feat[test_feat['TEAM'] == 'Auburn']['RANK'].values[0])
        duke = int(test_feat[test_feat['TEAM'] == 'Duke']['RANK'].values[0])
        
        print(f"{name:<22} {florida:>10} {houston:>10} {auburn:>10} {duke:>10}")
    
    print("\n*** Florida = 2025 ACTUAL CHAMPION ***")


if __name__ == "__main__":
    main()
