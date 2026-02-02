"""
Predict 2025 NCAA Tournament Champion
WITHOUT looking at actual results first
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.features.builder import FeatureBuilder
from src.models.champion_model import ChampionPredictor

# Load data
print('Loading data...')
df = pd.read_csv('data/raw/KenPom Barttorvik.csv')

# Get 2025 tournament teams (have a seed)
teams_2025 = df[(df['YEAR'] == 2025) & (df['SEED'].notna())].copy()
print(f'2025 tournament teams: {len(teams_2025)}')

# Get training data (all years before 2025)
train_df = df[(df['YEAR'] < 2025) & (df['YEAR'] >= 2008) & (df['SEED'].notna())].copy()

# Add SEED_NUM
train_df['SEED_NUM'] = train_df['SEED'].astype(float)
teams_2025['SEED_NUM'] = teams_2025['SEED'].astype(float)

# Create champion labels for training data (ROUND=1 means champion)
train_df['IS_CHAMPION'] = (train_df['ROUND'] == 1).astype(int)

champ_count = train_df['IS_CHAMPION'].sum()
print(f'Training on {len(train_df)} team-seasons with {champ_count} champions')

# Build features
builder = FeatureBuilder()
train_df_feat, X_train = builder.build_features(train_df, fit_scaler=True)
test_df_feat, X_test = builder.build_features(teams_2025, fit_scaler=False)

y_train = train_df_feat['IS_CHAMPION'].values

# Train model
print('Training model...')
model = ChampionPredictor(model_type='logreg')
model.fit(X_train, y_train, feature_names=builder.get_feature_names())

# Predict
probs = model.predict_proba(X_test)
test_df_feat = test_df_feat.copy()
test_df_feat['PROB'] = probs
test_df_feat['RANK'] = test_df_feat['PROB'].rank(ascending=False).astype(int)

# Show TOP 15 predictions
print()
print('='*60)
print('2025 NCAA CHAMPIONSHIP PREDICTIONS')
print('(Generated WITHOUT looking at actual results)')
print('='*60)
print()
print(f"{'Rank':<6}{'Team':<25}{'Seed':<6}{'Prob':<10}")
print('-'*50)

for _, row in test_df_feat.nlargest(15, 'PROB').iterrows():
    print(f"{int(row['RANK']):<6}{row['TEAM']:<25}{int(row['SEED']):<6}{row['PROB']:.3f}")

print()
top_pick = test_df_feat.nlargest(1, 'PROB').iloc[0]
print(f"*** MY PREDICTION: {top_pick['TEAM']} (Seed {int(top_pick['SEED'])}) ***")
print()

# Save predictions
test_df_feat[['RANK', 'TEAM', 'SEED', 'PROB']].sort_values('RANK').to_csv(
    'results/predictions_2025_blind.csv', index=False
)
print('Predictions saved to results/predictions_2025_blind.csv')
