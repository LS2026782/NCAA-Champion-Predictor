"""
Test the last remaining improvements:
1. TeamRankings CONSISTENCY (less variance = better tournament performer?)
2. Stacking ensemble (meta-model learns optimal weights)
3. Feature selection (remove noise)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import sys
sys.path.insert(0, '.')

from src.data.loader import DataLoader
from src.features.final_builder import FinalFeatureBuilder
from src.models.champion_model import ChampionPredictor
from config.settings import FIRST_TEST_YEAR

# =============================================================================
# 1. ANALYZE TEAMRANKINGS CONSISTENCY
# =============================================================================
print("="*70)
print("1. TEAMRANKINGS CONSISTENCY ANALYSIS")
print("="*70)

kp = pd.read_csv('data/raw/KenPom Barttorvik.csv')
kp = kp[(kp['YEAR'] >= 2008) & (kp['YEAR'] <= 2024) & (kp['SEED'].notna())].copy()
kp['IS_CHAMPION'] = (kp['ROUND'] == 1).astype(int)

tr = pd.read_csv('data/raw/TeamRankings.csv')

# Merge
merged = kp.merge(tr[['YEAR', 'TEAM', 'CONSISTENCY RANK', 'CONSISTENCY TR RATING', 'LUCK RATING']], 
                  on=['YEAR', 'TEAM'], how='left')

champs = merged[merged['IS_CHAMPION'] == 1]
field = merged[merged['IS_CHAMPION'] == 0]

print("\nTeamRankings features:")
for col in ['CONSISTENCY RANK', 'CONSISTENCY TR RATING', 'LUCK RATING']:
    c_mean = champs[col].mean()
    f_mean = field[col].mean()
    t, p = stats.ttest_ind(champs[col].dropna(), field[col].dropna())
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"  {col:<25}: Champs={c_mean:>7.1f}, Field={f_mean:>7.1f}, Diff={c_mean-f_mean:>+7.1f}, p={p:.4f} {sig}")

# =============================================================================
# 2. STACKING ENSEMBLE
# =============================================================================
print("\n" + "="*70)
print("2. STACKING ENSEMBLE (Meta-model learns optimal weights)")
print("="*70)

def run_stacking_backtest():
    """Train meta-model on base model predictions."""
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = sorted([y for y in loader.get_years() if y >= FIRST_TEST_YEAR])
    
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        # Get base model predictions on training data (using earlier years as "validation")
        # For simplicity, we'll use the predictions directly
        
        # Build features
        from src.features.builder import FeatureBuilder
        from src.features.ultimate_builder import UltimateFeatureBuilder
        
        # Train base models
        builders = [FeatureBuilder(), UltimateFeatureBuilder(), FinalFeatureBuilder()]
        base_train_preds = []
        base_test_preds = []
        
        for builder in builders:
            try:
                train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
                test_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
            except TypeError:
                train_feat, X_train = builder.build_features(train_df, fit_scaler=True)
                test_feat, X_test = builder.build_features(test_df, fit_scaler=False)
            
            y_train = train_feat['IS_CHAMPION'].values
            
            model = ChampionPredictor(model_type='logreg')
            model.fit(X_train, y_train, feature_names=builder.get_feature_names())
            
            train_probs = model.predict_proba(X_train)
            test_probs = model.predict_proba(X_test)
            
            base_train_preds.append(train_probs)
            base_test_preds.append(test_probs)
        
        # Stack: use base predictions as features for meta-model
        X_meta_train = np.column_stack(base_train_preds)
        X_meta_test = np.column_stack(base_test_preds)
        
        # Train meta-model (simple logistic regression)
        meta_model = LogisticRegressionCV(cv=3, class_weight='balanced', random_state=42, max_iter=1000)
        meta_model.fit(X_meta_train, y_train)
        
        # Get final predictions
        stacked_probs = meta_model.predict_proba(X_meta_test)[:, 1]
        
        test_feat = test_feat.copy()
        test_feat['PROB'] = stacked_probs
        test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
        
        champ = test_feat[test_feat['IS_CHAMPION'] == 1].iloc[0]
        results.append({
            'year': test_year,
            'champion': champ['TEAM'],
            'rank': int(champ['RANK'])
        })
        
        print(f"  {test_year}: {champ['TEAM']:<20} Rank={int(champ['RANK']):2d}")
    
    ranks = [r['rank'] for r in results]
    print(f"\nStacking Results:")
    print(f"  Mean Rank:  {np.mean(ranks):.2f}")
    print(f"  Top-5 Rate: {sum(r<=5 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"  Top-10 Rate: {sum(r<=10 for r in ranks)/len(ranks)*100:.1f}%")
    
    return results

stacking_results = run_stacking_backtest()

# =============================================================================
# 3. FEATURE SELECTION
# =============================================================================
print("\n" + "="*70)
print("3. FEATURE SELECTION (Top 15 features only)")
print("="*70)

def run_feature_selection_backtest():
    """Use only top 15 most important features."""
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = sorted([y for y in loader.get_years() if y >= FIRST_TEST_YEAR])
    
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        builder = FinalFeatureBuilder()
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
        test_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
        
        y_train = train_feat['IS_CHAMPION'].values
        
        # Select top 15 features
        selector = SelectKBest(f_classif, k=15)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Train model on selected features
        model = LogisticRegressionCV(cv=3, class_weight='balanced', random_state=42, max_iter=1000)
        model.fit(X_train_selected, y_train)
        
        probs = model.predict_proba(X_test_selected)[:, 1]
        
        test_feat = test_feat.copy()
        test_feat['PROB'] = probs
        test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
        
        champ = test_feat[test_feat['IS_CHAMPION'] == 1].iloc[0]
        results.append({
            'year': test_year,
            'champion': champ['TEAM'],
            'rank': int(champ['RANK'])
        })
        
        print(f"  {test_year}: {champ['TEAM']:<20} Rank={int(champ['RANK']):2d}")
    
    ranks = [r['rank'] for r in results]
    print(f"\nFeature Selection Results:")
    print(f"  Mean Rank:  {np.mean(ranks):.2f}")
    print(f"  Top-5 Rate: {sum(r<=5 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"  Top-10 Rate: {sum(r<=10 for r in ranks)/len(ranks)*100:.1f}%")
    
    return results

fs_results = run_feature_selection_backtest()

# =============================================================================
# FINAL COMPARISON
# =============================================================================
print("\n" + "="*70)
print("FINAL COMPARISON - New Techniques vs Best Model")
print("="*70)

print(f"\n{'Approach':<25} {'Mean Rank':>12} {'Top-5':>10} {'Top-10':>10}")
print("-"*60)

# Previous best (Final model)
print(f"{'Final (previous best)':<25} {'5.25':>12} {'66.7%':>10} {'83.3%':>10}")

# Stacking
s_ranks = [r['rank'] for r in stacking_results]
print(f"{'Stacking Ensemble':<25} {np.mean(s_ranks):>12.2f} {sum(r<=5 for r in s_ranks)/len(s_ranks)*100:>9.1f}% {sum(r<=10 for r in s_ranks)/len(s_ranks)*100:>9.1f}%")

# Feature Selection
f_ranks = [r['rank'] for r in fs_results]
print(f"{'Feature Selection (k=15)':<25} {np.mean(f_ranks):>12.2f} {sum(r<=5 for r in f_ranks)/len(f_ranks)*100:>9.1f}% {sum(r<=10 for r in f_ranks)/len(f_ranks)*100:>9.1f}%")
