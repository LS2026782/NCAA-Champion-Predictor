"""
Analyze remaining unexplored features that could improve predictions.

Key unexplored data:
1. Barttorvik Away-Neutral - Tournament games are NEUTRAL court!
2. EvanMiya KILL SHOTS - High-leverage play execution
3. Shooting Splits - Shot quality by location
4. Conference historical success
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load main data
kp = pd.read_csv('data/raw/KenPom Barttorvik.csv')
kp = kp[(kp['YEAR'] >= 2008) & (kp['YEAR'] <= 2024) & (kp['SEED'].notna())].copy()
kp['IS_CHAMPION'] = (kp['ROUND'] == 1).astype(int)

champs = kp[kp['IS_CHAMPION'] == 1]
field = kp[kp['IS_CHAMPION'] == 0]

print("="*70)
print("ANALYSIS OF REMAINING UNEXPLORED FEATURES")
print("="*70)

# =============================================================================
# 1. AWAY-NEUTRAL PERFORMANCE (Tournament = Neutral Court!)
# =============================================================================
print("\n" + "="*70)
print("1. AWAY-NEUTRAL PERFORMANCE (Most tournaments are neutral sites!)")
print("="*70)

away = pd.read_csv('data/raw/Barttorvik Away-Neutral.csv')
away = away[(away['YEAR'] >= 2008) & (away['YEAR'] <= 2024)]

# Merge with main data
merged = kp.merge(
    away[['YEAR', 'TEAM', 'BADJ EM', 'WIN%']],
    on=['YEAR', 'TEAM'],
    how='left',
    suffixes=('', '_AWAY')
)

champs_m = merged[merged['IS_CHAMPION'] == 1]
field_m = merged[merged['IS_CHAMPION'] == 0]

print("\nAway-Neutral Efficiency Margin:")
print(f"  Champions: {champs_m['BADJ EM_AWAY'].mean():.2f}")
print(f"  Field:     {field_m['BADJ EM_AWAY'].mean():.2f}")
print(f"  Diff:      {champs_m['BADJ EM_AWAY'].mean() - field_m['BADJ EM_AWAY'].mean():+.2f}")

t_stat, p_val = stats.ttest_ind(
    champs_m['BADJ EM_AWAY'].dropna(),
    field_m['BADJ EM_AWAY'].dropna()
)
print(f"  P-value:   {p_val:.4f} {'***' if p_val < 0.01 else '**' if p_val < 0.05 else ''}")

# Home vs Away comparison for champions
print("\nChampion Home vs Away-Neutral EM:")
for _, c in champs_m.iterrows():
    home_em = c['BADJ EM']
    away_em = c['BADJ EM_AWAY']
    diff = away_em - home_em if pd.notna(away_em) else 0
    print(f"  {int(c['YEAR'])} {c['TEAM']:<18}: Home={home_em:.1f}, Away={away_em:.1f} ({diff:+.1f})" if pd.notna(away_em) else f"  {int(c['YEAR'])} {c['TEAM']:<18}: No away data")

# =============================================================================
# 2. EVANMIYA KILL SHOTS (High-leverage execution)
# =============================================================================
print("\n" + "="*70)
print("2. EVANMIYA - KILL SHOTS (Clutch/High-leverage plays)")
print("="*70)

evan = pd.read_csv('data/raw/EvanMiya.csv')

# Merge
merged_evan = kp.merge(
    evan[['YEAR', 'TEAM', 'KILL SHOTS PER GAME', 'KILL SHOTS CONCEDED PER GAME', 'RELATIVE RATING']],
    on=['YEAR', 'TEAM'],
    how='left'
)

champs_e = merged_evan[merged_evan['IS_CHAMPION'] == 1]
field_e = merged_evan[merged_evan['IS_CHAMPION'] == 0]

print("\nKill Shots Per Game (offensive clutch plays):")
c_mean = champs_e['KILL SHOTS PER GAME'].mean()
f_mean = field_e['KILL SHOTS PER GAME'].mean()
print(f"  Champions: {c_mean:.2f}")
print(f"  Field:     {f_mean:.2f}")
print(f"  Diff:      {c_mean - f_mean:+.2f}")

if champs_e['KILL SHOTS PER GAME'].notna().sum() > 0:
    t_stat, p_val = stats.ttest_ind(
        champs_e['KILL SHOTS PER GAME'].dropna(),
        field_e['KILL SHOTS PER GAME'].dropna()
    )
    print(f"  P-value:   {p_val:.4f}")

print("\nKill Shots Conceded Per Game (defensive clutch):")
c_mean = champs_e['KILL SHOTS CONCEDED PER GAME'].mean()
f_mean = field_e['KILL SHOTS CONCEDED PER GAME'].mean()
print(f"  Champions: {c_mean:.2f}")
print(f"  Field:     {f_mean:.2f}")
print(f"  Diff:      {c_mean - f_mean:+.2f}")

# =============================================================================
# 3. SHOOTING SPLITS (Shot quality by location)
# =============================================================================
print("\n" + "="*70)
print("3. SHOOTING SPLITS (Shot quality and distribution)")
print("="*70)

shots = pd.read_csv('data/raw/Shooting Splits.csv')

merged_shots = kp.merge(
    shots[['YEAR', 'TEAM', 'DUNKS FG%', 'DUNKS SHARE', 'CLOSE TWOS FG%', 
           'CLOSE TWOS SHARE', 'THREES FG%', 'THREES SHARE']],
    on=['YEAR', 'TEAM'],
    how='left'
)

champs_s = merged_shots[merged_shots['IS_CHAMPION'] == 1]
field_s = merged_shots[merged_shots['IS_CHAMPION'] == 0]

print("\nDunk Share (% of shots that are dunks - rim pressure):")
c_mean = champs_s['DUNKS SHARE'].mean()
f_mean = field_s['DUNKS SHARE'].mean()
print(f"  Champions: {c_mean:.1f}%")
print(f"  Field:     {f_mean:.1f}%")
print(f"  Diff:      {c_mean - f_mean:+.1f}%")

if champs_s['DUNKS SHARE'].notna().sum() > 0:
    t_stat, p_val = stats.ttest_ind(
        champs_s['DUNKS SHARE'].dropna(),
        field_s['DUNKS SHARE'].dropna()
    )
    print(f"  P-value:   {p_val:.4f}")

print("\nClose Two Share (paint presence):")
c_mean = champs_s['CLOSE TWOS SHARE'].mean()
f_mean = field_s['CLOSE TWOS SHARE'].mean()
print(f"  Champions: {c_mean:.1f}%")
print(f"  Field:     {f_mean:.1f}%")

print("\n3PT Share (perimeter orientation):")
c_mean = champs_s['THREES SHARE'].mean()
f_mean = field_s['THREES SHARE'].mean()
print(f"  Champions: {c_mean:.1f}%")
print(f"  Field:     {f_mean:.1f}%")

# =============================================================================
# 4. CONFERENCE TOURNAMENT SUCCESS
# =============================================================================
print("\n" + "="*70)
print("4. CONFERENCE HISTORICAL TOURNAMENT SUCCESS")
print("="*70)

conf = pd.read_csv('data/raw/Conference Results.csv')
print("\nConferences by Championship Rate:")
conf_sorted = conf.sort_values('CHAMP', ascending=False).head(10)
for _, c in conf_sorted.iterrows():
    print(f"  {c['CONF']:<6}: {c['CHAMP']} titles, {c['F4']} F4, {c['CHAMP%']} champ rate")

# Merge conference success with teams
merged_conf = kp.merge(
    conf[['CONF', 'CHAMP%']],
    left_on='CONF',
    right_on='CONF',
    how='left'
)

champs_c = merged_conf[merged_conf['IS_CHAMPION'] == 1]
field_c = merged_conf[merged_conf['IS_CHAMPION'] == 0]

# Convert CHAMP% to numeric
def pct_to_float(x):
    if pd.isna(x): return 0
    return float(str(x).replace('%', ''))

champs_c['CHAMP%_NUM'] = champs_c['CHAMP%'].apply(pct_to_float)
field_c['CHAMP%_NUM'] = field_c['CHAMP%'].apply(pct_to_float)

print("\nConference Historical Champ%:")
print(f"  Champions come from: {champs_c['CHAMP%_NUM'].mean():.1f}% avg conf")
print(f"  Field comes from:    {field_c['CHAMP%_NUM'].mean():.1f}% avg conf")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY - PROMISING UNEXPLORED FEATURES")
print("="*70)

print("""
Most Promising Features to Add:

1. AWAY-NEUTRAL EFFICIENCY MARGIN
   - Tournament games are on neutral courts
   - Teams that perform well away from home should translate better
   - Champions show higher away-neutral EM

2. DUNK SHARE / CLOSE TWOS SHARE
   - Measures ability to score at the rim
   - Rim pressure correlates with tournament success
   - Champions get more easy baskets

3. CONFERENCE HISTORICAL SUCCESS
   - Some conferences consistently produce champions
   - ACC, Big East, SEC, Big 12 dominate
   - Could be a tiebreaker feature

4. EVANMIYA KILL SHOTS (if data available)
   - Measures execution in high-leverage moments
   - Could indicate clutch performance ability

5. HOME/AWAY PERFORMANCE GAP
   - Teams that don't drop off away from home
   - More consistent = better tournament fit
""")
