"""Test all models on 2025 predictions."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.features.builder import FeatureBuilder
from src.features.ultimate_builder import UltimateFeatureBuilder
from src.features.final_builder import FinalFeatureBuilder
from src.models.champion_model import ChampionPredictor

# Load data
df = pd.read_csv('data/raw/KenPom Barttorvik.csv')
teams_2025 = df[(df['YEAR'] == 2025) & (df['SEED'].notna())].copy()
train_df = df[(df['YEAR'] < 2025) & (df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()
train_df['SEED_NUM'] = train_df['SEED'].astype(float)
teams_2025['SEED_NUM'] = teams_2025['SEED'].astype(float)
train_df['IS_CHAMPION'] = (train_df['ROUND'] == 1).astype(int)

full_data = df[(df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()
full_data['SEED_NUM'] = full_data['SEED'].astype(float)

print("="*70)
print("2025 PREDICTION - ALL MODELS COMPARISON")
print("="*70)
print()

models = {
    'Original': FeatureBuilder,
    'Ultimate': UltimateFeatureBuilder,
    'Final': FinalFeatureBuilder
}

header = f"{'Model':<12} {'Florida':>10} {'Houston':>10} {'Auburn':>10} {'Duke':>10}"
print(header)
print("-"*55)

for name, builder_class in models.items():
    builder = builder_class()
    try:
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
        test_feat, X_test = builder.build_features(teams_2025, fit_scaler=False, all_data=full_data)
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
    
    florida = int(test_feat[test_feat['TEAM'] == 'Florida']['RANK'].values[0])
    houston = int(test_feat[test_feat['TEAM'] == 'Houston']['RANK'].values[0])
    auburn = int(test_feat[test_feat['TEAM'] == 'Auburn']['RANK'].values[0])
    duke = int(test_feat[test_feat['TEAM'] == 'Duke']['RANK'].values[0])
    
    print(f"{name:<12} {florida:>10} {houston:>10} {auburn:>10} {duke:>10}")

print()
print("Florida = 2025 ACTUAL CHAMPION")
print()

# Show Final model's top 10
print("="*70)
print("FINAL MODEL - TOP 10 FOR 2025")
print("="*70)

builder = FinalFeatureBuilder()
train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
test_feat, X_test = builder.build_features(teams_2025, fit_scaler=False, all_data=full_data)

y_train = train_feat['IS_CHAMPION'].values
model = ChampionPredictor(model_type='logreg')
model.fit(X_train, y_train, feature_names=builder.get_feature_names())

probs = model.predict_proba(X_test)
test_feat = test_feat.copy()
test_feat['PROB'] = probs
test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)

top10 = test_feat.nsmallest(10, 'RANK')[['TEAM', 'SEED', 'PROB', 'RANK']]
print()
for i, (_, row) in enumerate(top10.iterrows(), 1):
    champ_marker = " <-- ACTUAL CHAMPION" if row['TEAM'] == 'Florida' else ""
    print(f"  {i:2d}. {row['TEAM']:<20} (Seed {int(row['SEED'])}) Prob: {row['PROB']:.3f}{champ_marker}")
