"""
Predict 2025 with enhanced model and compare to original.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.features.builder import FeatureBuilder
from src.features.enhanced_builder import EnhancedFeatureBuilder  
from src.models.champion_model import ChampionPredictor

# Load data
print("Loading data...")
df = pd.read_csv('data/raw/KenPom Barttorvik.csv')

# Get 2025 tournament teams
teams_2025 = df[(df['YEAR'] == 2025) & (df['SEED'].notna())].copy()
print(f'2025 tournament teams: {len(teams_2025)}')

# Get training data (all years before 2025)
train_df = df[(df['YEAR'] < 2025) & (df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()

# Add required columns
train_df['SEED_NUM'] = train_df['SEED'].astype(float)
teams_2025['SEED_NUM'] = teams_2025['SEED'].astype(float)
train_df['IS_CHAMPION'] = (train_df['ROUND'] == 1).astype(int)
teams_2025['IS_CHAMPION'] = 0  # We don't know yet

print(f'Training on {len(train_df)} team-seasons')

# Full data for ranks
full_data = df[(df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()
full_data['SEED_NUM'] = full_data['SEED'].astype(float)

# ========== ORIGINAL MODEL ==========
print("\n[1/2] Training ORIGINAL model...")
orig_builder = FeatureBuilder()
train_orig, X_train_orig = orig_builder.build_features(train_df, fit_scaler=True)
test_orig, X_test_orig = orig_builder.build_features(teams_2025, fit_scaler=False)

y_train = train_orig['IS_CHAMPION'].values

orig_model = ChampionPredictor(model_type='logreg')
orig_model.fit(X_train_orig, y_train, feature_names=orig_builder.get_feature_names())

probs_orig = orig_model.predict_proba(X_test_orig)

# ========== ENHANCED MODEL ==========
print("\n[2/2] Training ENHANCED model...")
enh_builder = EnhancedFeatureBuilder()
train_enh, X_train_enh = enh_builder.build_features(train_df, fit_scaler=True, all_data=full_data)
test_enh, X_test_enh = enh_builder.build_features(teams_2025, fit_scaler=False, all_data=full_data)

enh_model = ChampionPredictor(model_type='logreg')
enh_model.fit(X_train_enh, y_train, feature_names=enh_builder.get_feature_names())

probs_enh = enh_model.predict_proba(X_test_enh)

# ========== RESULTS ==========
results = teams_2025[['TEAM', 'SEED']].copy()
results['PROB_ORIGINAL'] = probs_orig
results['PROB_ENHANCED'] = probs_enh
results['RANK_ORIGINAL'] = results['PROB_ORIGINAL'].rank(ascending=False).astype(int)
results['RANK_ENHANCED'] = results['PROB_ENHANCED'].rank(ascending=False).astype(int)

# Ensemble (60% enhanced, 40% original based on median performance)
results['PROB_ENSEMBLE'] = 0.4 * probs_orig + 0.6 * probs_enh
results['RANK_ENSEMBLE'] = results['PROB_ENSEMBLE'].rank(ascending=False).astype(int)

print("\n" + "="*70)
print("2025 PREDICTIONS: ORIGINAL vs ENHANCED vs ENSEMBLE")
print("="*70)
print(f"\n{'Team':<20} {'Seed':>5} {'Orig':>8} {'Enh':>8} {'Ens':>8}")
print("-"*55)

for _, row in results.nlargest(15, 'PROB_ENSEMBLE').iterrows():
    champ_marker = " <-- ACTUAL CHAMPION" if row['TEAM'] == 'Florida' else ""
    print(f"{row['TEAM']:<20} {int(row['SEED']):>5} "
          f"{int(row['RANK_ORIGINAL']):>8} {int(row['RANK_ENHANCED']):>8} "
          f"{int(row['RANK_ENSEMBLE']):>8}{champ_marker}")

# Find Florida specifically
florida = results[results['TEAM'] == 'Florida'].iloc[0]
print("\n" + "="*70)
print("FLORIDA (ACTUAL 2025 CHAMPION) RANKINGS")
print("="*70)
print(f"Original Model:  #{int(florida['RANK_ORIGINAL'])} (prob: {florida['PROB_ORIGINAL']:.3f})")
print(f"Enhanced Model:  #{int(florida['RANK_ENHANCED'])} (prob: {florida['PROB_ENHANCED']:.3f})")
print(f"Ensemble Model:  #{int(florida['RANK_ENSEMBLE'])} (prob: {florida['PROB_ENSEMBLE']:.3f})")

# Also check Houston (runner-up)
houston = results[results['TEAM'] == 'Houston'].iloc[0]
print(f"\nHouston (Runner-up) Rankings:")
print(f"Original: #{int(houston['RANK_ORIGINAL'])}, Enhanced: #{int(houston['RANK_ENHANCED'])}, Ensemble: #{int(houston['RANK_ENSEMBLE'])}")
