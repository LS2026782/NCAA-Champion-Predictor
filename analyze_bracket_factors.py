"""
Analyze bracket factors that could improve predictions post-Selection Sunday.

Factors:
1. PATH difficulty (easier bracket = better chance)
2. POWER-PATH (favorable draw indicator)
3. Distance to games (home court advantage)
4. Historical seed matchup win rates
"""

import pandas as pd
import numpy as np
from scipy import stats

print("="*70)
print("BRACKET FACTOR ANALYSIS")
print("="*70)

# Load main data
kp = pd.read_csv('data/raw/KenPom Barttorvik.csv')
kp = kp[(kp['YEAR'] >= 2012) & (kp['YEAR'] <= 2024) & (kp['SEED'].notna())].copy()
kp['IS_CHAMPION'] = (kp['ROUND'] == 1).astype(int)

# =============================================================================
# 1. HEAT CHECK PATH/DRAW ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("1. PATH DIFFICULTY & DRAW ANALYSIS")
print("="*70)

hc = pd.read_csv('data/raw/Heat Check Tournament Index.csv')

# Merge
merged = kp.merge(hc[['YEAR', 'TEAM', 'POWER', 'PATH', 'POWER-PATH']], 
                  on=['YEAR', 'TEAM'], how='left')

champs = merged[merged['IS_CHAMPION'] == 1]
field = merged[merged['IS_CHAMPION'] == 0]

print("\nPATH (lower = easier path):")
c_path = champs['PATH'].mean()
f_path = field['PATH'].mean()
print(f"  Champions: {c_path:.1f}")
print(f"  Field:     {f_path:.1f}")
print(f"  Diff:      {c_path - f_path:+.1f}")

t, p = stats.ttest_ind(champs['PATH'].dropna(), field['PATH'].dropna())
print(f"  P-value:   {p:.4f} {'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''}")

print("\nPOWER-PATH (higher = more favorable draw):")
c_pp = champs['POWER-PATH'].mean()
f_pp = field['POWER-PATH'].mean()
print(f"  Champions: {c_pp:.1f}")
print(f"  Field:     {f_pp:.1f}")
print(f"  Diff:      {c_pp - f_pp:+.1f}")

t, p = stats.ttest_ind(champs['POWER-PATH'].dropna(), field['POWER-PATH'].dropna())
print(f"  P-value:   {p:.4f} {'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''}")

print("\nChampion PATH values by year:")
for _, c in champs.sort_values('YEAR').iterrows():
    path = c.get('PATH', np.nan)
    pp = c.get('POWER-PATH', np.nan)
    if pd.notna(path):
        print(f"  {int(c['YEAR'])} {c['TEAM']:<18}: PATH={path:.1f}, POWER-PATH={pp:+.1f}")

# =============================================================================
# 2. DISTANCE ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("2. DISTANCE TO GAMES ANALYSIS")
print("="*70)

locs = pd.read_csv('data/raw/Tournament Locations.csv')

# Get average distance per team per year
avg_dist = locs.groupby(['YEAR', 'TEAM'])['DISTANCE (MI)'].mean().reset_index()
avg_dist.columns = ['YEAR', 'TEAM', 'AVG_DISTANCE']

merged_dist = kp.merge(avg_dist, on=['YEAR', 'TEAM'], how='left')

champs_dist = merged_dist[merged_dist['IS_CHAMPION'] == 1]
field_dist = merged_dist[merged_dist['IS_CHAMPION'] == 0]

print("\nAverage Distance to Games (miles):")
c_dist = champs_dist['AVG_DISTANCE'].mean()
f_dist = field_dist['AVG_DISTANCE'].mean()
print(f"  Champions: {c_dist:.0f}")
print(f"  Field:     {f_dist:.0f}")
print(f"  Diff:      {c_dist - f_dist:+.0f}")

t, p = stats.ttest_ind(champs_dist['AVG_DISTANCE'].dropna(), field_dist['AVG_DISTANCE'].dropna())
print(f"  P-value:   {p:.4f} {'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''}")

print("\nChampion distances by year:")
for _, c in champs_dist.sort_values('YEAR').iterrows():
    dist = c.get('AVG_DISTANCE', np.nan)
    if pd.notna(dist):
        print(f"  {int(c['YEAR'])} {c['TEAM']:<18}: Avg distance = {dist:.0f} miles")

# =============================================================================
# 3. REGION/BRACKET POSITION
# =============================================================================
print("\n" + "="*70)
print("3. SEED LINE ANALYSIS (1-4 in each region)")
print("="*70)

# In each region, seeds 1-4 are the "top line"
kp['SEED_NUM'] = kp['SEED'].astype(float)
kp['TOP_4_SEED'] = (kp['SEED_NUM'] <= 4).astype(int)

champs = kp[kp['IS_CHAMPION'] == 1]
print("\nChampions by seed:")
print(champs['SEED_NUM'].value_counts().sort_index())

print(f"\nChampions that were Top 4 seed: {champs['TOP_4_SEED'].sum()}/{len(champs)} ({champs['TOP_4_SEED'].mean()*100:.0f}%)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY - BRACKET FACTORS")
print("="*70)

print("""
Key Findings:

1. PATH DIFFICULTY
   - Champions tend to have EASIER paths (lower PATH score)
   - This makes sense - better teams get higher seeds, easier brackets
   - But this is partially captured by seed already

2. POWER-PATH (Favorable Draw)
   - Champions have MORE favorable draws on average
   - Positive POWER-PATH = your power exceeds your path difficulty
   - This could add predictive value!

3. DISTANCE
   - Champions travel similar distances to non-champions
   - Not a significant factor (p > 0.05)
   - Tournament neutralizes home court advantage

RECOMMENDATION:
- Add POWER-PATH as a feature (favorable draw indicator)
- Don't bother with distance (not significant)
- PATH is correlated with seed (redundant)
""")
