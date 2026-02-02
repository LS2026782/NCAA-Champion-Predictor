"""Test Momentum model on 2025"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.features.momentum_builder import MomentumFeatureBuilder
from src.models.champion_model import ChampionPredictor

# Load data
print("Loading data...")
df = pd.read_csv('data/raw/KenPom Barttorvik.csv')

# Get 2025 tournament teams
teams_2025 = df[(df['YEAR'] == 2025) & (df['SEED'].notna())].copy()

# Get training data
train_df = df[(df['YEAR'] < 2025) & (df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()

# Add required columns
train_df['SEED_NUM'] = train_df['SEED'].astype(float)
teams_2025['SEED_NUM'] = teams_2025['SEED'].astype(float)
train_df['IS_CHAMPION'] = (train_df['ROUND'] == 1).astype(int)

# Full data for reference
full_data = df[(df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()
full_data['SEED_NUM'] = full_data['SEED'].astype(float)

print(f"Training: {len(train_df)} teams, Testing: {len(teams_2025)} teams")

# Build momentum features
print("\nBuilding momentum features...")
builder = MomentumFeatureBuilder()
train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
test_feat, X_test = builder.build_features(teams_2025, fit_scaler=False, all_data=full_data)

y_train = train_feat['IS_CHAMPION'].values

# Train
print("Training model...")
model = ChampionPredictor(model_type='logreg')
model.fit(X_train, y_train, feature_names=builder.get_feature_names())

# Predict
probs = model.predict_proba(X_test)
test_feat = test_feat.copy()
test_feat['PROB'] = probs
test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)

print("\n" + "="*70)
print("2025 MOMENTUM MODEL PREDICTIONS")
print("="*70)

print(f"\n{'Rank':<6}{'Team':<22}{'Seed':>5}{'Mom':>8}{'Q1 W':>7}{'Prob':>10}")
print("-"*60)

for _, row in test_feat.nlargest(15, 'PROB').iterrows():
    mom = row.get('MOMENTUM_EM', 0)
    q1 = row.get('Q1_WINS', 0)
    marker = " <-- CHAMPION" if row['TEAM'] == 'Florida' else ""
    print(f"{int(row['RANK']):<6}{row['TEAM']:<22}{int(row['SEED']):>5}{mom:>+8.1f}{int(q1):>7}{row['PROB']:>10.3f}{marker}")

# Florida's position
florida = test_feat[test_feat['TEAM'] == 'Florida'].iloc[0]
houston = test_feat[test_feat['TEAM'] == 'Houston'].iloc[0]

print("\n" + "="*70)
print("2025 FINALS MATCHUP")
print("="*70)
print(f"Florida (Champion):  Rank #{int(florida['RANK'])}, Momentum: {florida.get('MOMENTUM_EM', 0):+.1f}, Q1 Wins: {int(florida.get('Q1_WINS', 0))}")
print(f"Houston (Runner-up): Rank #{int(houston['RANK'])}, Momentum: {houston.get('MOMENTUM_EM', 0):+.1f}, Q1 Wins: {int(houston.get('Q1_WINS', 0))}")
