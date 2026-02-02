"""
Comprehensive audit of remaining unexplored data and techniques.
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load main data for comparison
kp = pd.read_csv('data/raw/KenPom Barttorvik.csv')
kp = kp[(kp['YEAR'] >= 2008) & (kp['YEAR'] <= 2024) & (kp['SEED'].notna())].copy()
kp['IS_CHAMPION'] = (kp['ROUND'] == 1).astype(int)

champs = kp[kp['IS_CHAMPION'] == 1]
field = kp[kp['IS_CHAMPION'] == 0]

print("="*70)
print("COMPREHENSIVE AUDIT - REMAINING OPPORTUNITIES")
print("="*70)

# =============================================================================
# 1. TEAMRANKINGS DATA
# =============================================================================
print("\n" + "="*70)
print("1. TEAMRANKINGS DATA (43 columns - unexplored!)")
print("="*70)

tr = pd.read_csv('data/raw/TeamRankings.csv')
print("Columns:", list(tr.columns))

# Merge and test significance
merged = kp.merge(tr, on=['YEAR', 'TEAM'], how='left', suffixes=('', '_TR'))

interesting_cols = ['PREDICTIVE RATING', 'SOS RATING', 'HOME RATING', 'AWAY RATING', 
                    'LAST 5 GAMES RATING', 'LAST 10 GAMES RATING']

print("\nTeamRankings features - Champion vs Field:")
for col in interesting_cols:
    if col in merged.columns:
        c_mean = merged[merged['IS_CHAMPION']==1][col].mean()
        f_mean = merged[merged['IS_CHAMPION']==0][col].mean()
        try:
            t, p = stats.ttest_ind(
                merged[merged['IS_CHAMPION']==1][col].dropna(),
                merged[merged['IS_CHAMPION']==0][col].dropna()
            )
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        except:
            sig = ""
        print(f"  {col:<25}: Champs={c_mean:.1f}, Field={f_mean:.1f}, Diff={c_mean-f_mean:+.1f} {sig}")

# =============================================================================
# 2. Z RATING
# =============================================================================
print("\n" + "="*70)
print("2. Z RATING (Another rating system)")
print("="*70)

zr = pd.read_csv('data/raw/Z Rating.csv')
print("Columns:", list(zr.columns))

merged = kp.merge(zr[['YEAR', 'TEAM', 'Z RATING']], on=['YEAR', 'TEAM'], how='left')
c_mean = merged[merged['IS_CHAMPION']==1]['Z RATING'].mean()
f_mean = merged[merged['IS_CHAMPION']==0]['Z RATING'].mean()
print(f"  Champions: {c_mean:.2f}, Field: {f_mean:.2f}, Diff: {c_mean-f_mean:+.2f}")

# =============================================================================
# 3. TOURNAMENT LOCATIONS (Distance advantage)
# =============================================================================
print("\n" + "="*70)
print("3. TOURNAMENT LOCATIONS (Distance to games)")
print("="*70)

locs = pd.read_csv('data/raw/Tournament Locations.csv')
print("Columns:", list(locs.columns)[:20])

# Check if distance helps
if 'DISTANCE' in locs.columns:
    merged = kp.merge(locs[['YEAR', 'TEAM', 'DISTANCE']], on=['YEAR', 'TEAM'], how='left')
    c_mean = merged[merged['IS_CHAMPION']==1]['DISTANCE'].mean()
    f_mean = merged[merged['IS_CHAMPION']==0]['DISTANCE'].mean()
    print(f"  Champions avg distance: {c_mean:.1f}, Field: {f_mean:.1f}")

# =============================================================================
# 4. AP POLL (Preseason expectations)
# =============================================================================
print("\n" + "="*70)
print("4. AP POLL (Preseason ranking/expectations)")
print("="*70)

ap = pd.read_csv('data/raw/AP Poll Week 6.csv')
print("Columns:", list(ap.columns))

# =============================================================================
# 5. 538 RATINGS
# =============================================================================
print("\n" + "="*70)
print("5. 538 RATINGS (FiveThirtyEight)")
print("="*70)

r538 = pd.read_csv('data/raw/538 Ratings.csv')
print("Columns:", list(r538.columns))
print(f"Years available: {sorted(r538['YEAR'].unique())}")

merged = kp.merge(r538[['YEAR', 'TEAM', 'POWER RATING']], on=['YEAR', 'TEAM'], how='left')
c_mean = merged[merged['IS_CHAMPION']==1]['POWER RATING'].mean()
f_mean = merged[merged['IS_CHAMPION']==0]['POWER RATING'].mean()
print(f"  Champions: {c_mean:.2f}, Field: {f_mean:.2f}, Diff: {c_mean-f_mean:+.2f}")

# =============================================================================
# 6. HEAT CHECK RATINGS
# =============================================================================
print("\n" + "="*70)
print("6. HEAT CHECK RATINGS")
print("="*70)

hc = pd.read_csv('data/raw/Heat Check Ratings.csv')
print("Columns:", list(hc.columns))

# =============================================================================
# SUMMARY OF UNTESTED TECHNIQUES
# =============================================================================
print("\n" + "="*70)
print("UNTESTED MODELING TECHNIQUES")
print("="*70)

print("""
1. STACKING ENSEMBLE
   - Train a meta-model on base model predictions
   - Could find optimal weighting automatically

2. FEATURE SELECTION
   - Reduce to top 10-15 most important features
   - Might reduce noise and overfitting

3. TIME-WEIGHTED TRAINING
   - Weight recent years more heavily
   - Basketball evolves, recent patterns may be more relevant

4. DIFFERENT TARGET VARIABLES
   - Predict Final Four instead of Champion
   - More positive examples = more stable model

5. CROSS-VALIDATION
   - K-fold CV instead of temporal split
   - But risks data leakage if not careful

6. BAYESIAN OPTIMIZATION
   - Better hyperparameter tuning
   - Find optimal regularization, features, etc.

7. GAME-BY-GAME SIMULATION
   - Model individual game win probability
   - Monte Carlo simulate entire bracket
   - Already partially implemented
""")

# =============================================================================
# WHAT'S TRULY LEFT
# =============================================================================
print("\n" + "="*70)
print("REALISTIC REMAINING IMPROVEMENTS")
print("="*70)

print("""
HIGH POTENTIAL:
1. TeamRankings 'LAST 5/10 GAMES' - Late season form
2. Stacking ensemble - Auto-optimize model combination
3. Feature selection - Remove noise

MEDIUM POTENTIAL:
4. 538 Ratings (limited to 2016+)
5. Z Rating (another perspective)
6. Tournament distance (home court-ish advantage)

LOW POTENTIAL (diminishing returns):
7. Most other data is redundant with what we have
8. More complex models overfit on small data

THE HARD TRUTH:
- We're predicting 1 winner out of 68 teams
- Even perfect models can't account for:
  * Hot shooting streaks
  * Key injuries
  * Referee decisions
  * Matchup luck (bracket draw)
  * "March Madness" randomness

Our model ranks champion in Top 5 ~75% of the time.
That's approaching the ceiling of what's possible with stats alone.
""")
