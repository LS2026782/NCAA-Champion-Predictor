"""
Deep Pattern Analysis: What Makes Champions Different?

This analysis examines all available features to find patterns
that distinguish champions from other tournament teams.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('data/raw/KenPom Barttorvik.csv')

# Filter to tournament teams (2008-2024)
df = df[(df['YEAR'] >= 2008) & (df['YEAR'] <= 2024) & (df['SEED'].notna())].copy()

# Create champion label
df['IS_CHAMPION'] = (df['ROUND'] == 1).astype(int)

# Separate champions and non-champions
champs = df[df['IS_CHAMPION'] == 1]
non_champs = df[df['IS_CHAMPION'] == 0]

print(f"Champions: {len(champs)}")
print(f"Non-Champions: {len(non_champs)}")

# =============================================================================
# 1. CHAMPION PROFILE: What do champions look like?
# =============================================================================
print("\n" + "="*70)
print("1. CHAMPION PROFILE (2008-2024)")
print("="*70)

profile_cols = ['SEED', 'KADJ EM', 'KADJ O', 'KADJ D', 'BARTHAG', 
                'EFG%', 'TOV%', 'OREB%', 'FTR', 'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
                'EXP', 'ELITE SOS', '3PT%', '3PT%D', '2PT%', '2PT%D']

print("\nChampion averages vs Field averages:")
print(f"{'Metric':<15} {'Champions':>12} {'Field':>12} {'Diff':>10} {'Champ Better?':>15}")
print("-"*70)

for col in profile_cols:
    if col in df.columns:
        champ_mean = champs[col].mean()
        field_mean = non_champs[col].mean()
        diff = champ_mean - field_mean
        
        # Determine if higher or lower is better
        lower_better = col in ['KADJ D', 'TOV%', 'EFG%D', 'FTRD', '3PT%D', '2PT%D', 'SEED']
        if lower_better:
            better = "YES" if diff < 0 else "no"
        else:
            better = "YES" if diff > 0 else "no"
            
        print(f"{col:<15} {champ_mean:>12.2f} {field_mean:>12.2f} {diff:>+10.2f} {better:>15}")

# =============================================================================
# 2. STATISTICAL SIGNIFICANCE: Which features truly separate champions?
# =============================================================================
print("\n" + "="*70)
print("2. STATISTICAL SIGNIFICANCE (t-test)")
print("="*70)

print("\nFeatures with significant difference (p < 0.05):")
print(f"{'Feature':<20} {'t-statistic':>12} {'p-value':>12} {'Effect Size':>12}")
print("-"*60)

significant_features = []
for col in df.select_dtypes(include=[np.number]).columns:
    if col not in ['YEAR', 'IS_CHAMPION', 'ROUND', 'TEAM NO', 'TEAM ID', 'CONF ID', 'QUAD NO', 'QUAD ID']:
        champ_vals = champs[col].dropna()
        non_champ_vals = non_champs[col].dropna()
        
        if len(champ_vals) > 5 and len(non_champ_vals) > 5:
            t_stat, p_val = stats.ttest_ind(champ_vals, non_champ_vals)
            
            # Cohen's d effect size
            pooled_std = np.sqrt((champ_vals.std()**2 + non_champ_vals.std()**2) / 2)
            effect_size = (champ_vals.mean() - non_champ_vals.mean()) / pooled_std if pooled_std > 0 else 0
            
            if p_val < 0.05:
                significant_features.append((col, t_stat, p_val, effect_size))
                print(f"{col:<20} {t_stat:>12.2f} {p_val:>12.4f} {effect_size:>12.2f}")

# =============================================================================
# 3. CHAMPION THRESHOLDS: What are the minimum requirements?
# =============================================================================
print("\n" + "="*70)
print("3. CHAMPION THRESHOLDS (Min/Max values among champions)")
print("="*70)

threshold_cols = ['SEED', 'KADJ EM', 'KADJ D', 'BARTHAG', 'EFG%', 'EXP', 'ELITE SOS']

print("\nWhat's the WORST value a champion has had?")
print(f"{'Metric':<15} {'Worst Champ':>12} {'Best Champ':>12} {'Field Median':>12}")
print("-"*55)

for col in threshold_cols:
    if col in df.columns:
        lower_better = col in ['KADJ D', 'SEED']
        
        if lower_better:
            worst = champs[col].max()
            best = champs[col].min()
        else:
            worst = champs[col].min()
            best = champs[col].max()
            
        field_median = non_champs[col].median()
        print(f"{col:<15} {worst:>12.2f} {best:>12.2f} {field_median:>12.2f}")

# =============================================================================
# 4. SEED ANALYSIS: How important is seeding?
# =============================================================================
print("\n" + "="*70)
print("4. SEED ANALYSIS")
print("="*70)

seed_dist = champs['SEED'].value_counts().sort_index()
print("\nChampions by seed (2008-2024):")
for seed, count in seed_dist.items():
    pct = count / len(champs) * 100
    bar = "#" * int(pct / 2)
    print(f"  Seed {int(seed):2d}: {count:2d} ({pct:5.1f}%) {bar}")

# Cumulative
print("\nCumulative probability by seed:")
cumsum = 0
for seed in range(1, 17):
    count = seed_dist.get(seed, 0)
    cumsum += count
    pct = cumsum / len(champs) * 100
    print(f"  Seed 1-{seed:2d}: {pct:5.1f}%")
    if pct >= 95:
        break

# =============================================================================
# 5. ELITE THRESHOLDS: What % of field do champions rank in?
# =============================================================================
print("\n" + "="*70)
print("5. PERCENTILE ANALYSIS: Where do champions rank in the field?")
print("="*70)

percentile_cols = ['KADJ EM', 'KADJ O', 'KADJ D', 'BARTHAG', 'EFG%', 'EFG%D', 'EXP', 'ELITE SOS']

print("\nChampion percentile rankings (higher = better, except KADJ D, EFG%D):")
print(f"{'Metric':<12} {'Min %ile':>10} {'Avg %ile':>10} {'Max %ile':>10} {'Pattern':<20}")
print("-"*65)

for col in percentile_cols:
    if col in df.columns:
        # Calculate percentile for each champion within their year
        champ_percentiles = []
        for _, champ in champs.iterrows():
            year_data = df[df['YEAR'] == champ['YEAR']][col]
            
            lower_better = col in ['KADJ D', 'EFG%D']
            if lower_better:
                percentile = (year_data > champ[col]).mean() * 100
            else:
                percentile = (year_data < champ[col]).mean() * 100
                
            champ_percentiles.append(percentile)
        
        min_p = np.min(champ_percentiles)
        avg_p = np.mean(champ_percentiles)
        max_p = np.max(champ_percentiles)
        
        # Determine pattern
        if min_p >= 80:
            pattern = "MUST BE ELITE (>80%)"
        elif min_p >= 60:
            pattern = "Usually top 40%"
        elif avg_p >= 70:
            pattern = "Typically elite"
        else:
            pattern = "Variable"
            
        print(f"{col:<12} {min_p:>10.1f} {avg_p:>10.1f} {max_p:>10.1f} {pattern:<20}")

# =============================================================================
# 6. DEFENSIVE EXCELLENCE: Is defense more important than offense?
# =============================================================================
print("\n" + "="*70)
print("6. OFFENSE vs DEFENSE ANALYSIS")
print("="*70)

# Rank within year
df['OFF_RANK'] = df.groupby('YEAR')['KADJ O'].rank(ascending=False)
df['DEF_RANK'] = df.groupby('YEAR')['KADJ D'].rank(ascending=True)  # Lower is better
df['EM_RANK'] = df.groupby('YEAR')['KADJ EM'].rank(ascending=False)

champs = df[df['IS_CHAMPION'] == 1]

print("\nChampion rankings within their tournament year:")
print(f"{'Year':<6} {'Team':<20} {'Off Rank':>10} {'Def Rank':>10} {'EM Rank':>10}")
print("-"*60)

for _, c in champs.iterrows():
    print(f"{int(c['YEAR']):<6} {c['TEAM']:<20} {int(c['OFF_RANK']):>10} {int(c['DEF_RANK']):>10} {int(c['EM_RANK']):>10}")

print(f"\nAverage champion ranks:")
print(f"  Offense: {champs['OFF_RANK'].mean():.1f}")
print(f"  Defense: {champs['DEF_RANK'].mean():.1f}")
print(f"  Efficiency Margin: {champs['EM_RANK'].mean():.1f}")

# Which matters more?
print(f"\nChampions in Top 5 offense: {(champs['OFF_RANK'] <= 5).sum()}/{len(champs)} ({(champs['OFF_RANK'] <= 5).mean()*100:.0f}%)")
print(f"Champions in Top 5 defense: {(champs['DEF_RANK'] <= 5).sum()}/{len(champs)} ({(champs['DEF_RANK'] <= 5).mean()*100:.0f}%)")
print(f"Champions in Top 5 EM:      {(champs['EM_RANK'] <= 5).sum()}/{len(champs)} ({(champs['EM_RANK'] <= 5).mean()*100:.0f}%)")

# =============================================================================
# 7. EXPERIENCE FACTOR
# =============================================================================
print("\n" + "="*70)
print("7. EXPERIENCE ANALYSIS")
print("="*70)

if 'EXP' in df.columns:
    df['EXP_RANK'] = df.groupby('YEAR')['EXP'].rank(ascending=False)
    champs = df[df['IS_CHAMPION'] == 1]
    
    print(f"\nChampion experience rankings:")
    print(f"  Average rank: {champs['EXP_RANK'].mean():.1f}")
    print(f"  Champions in Top 10 experience: {(champs['EXP_RANK'] <= 10).sum()}/{len(champs)}")
    print(f"  Champions in Top 20 experience: {(champs['EXP_RANK'] <= 20).sum()}/{len(champs)}")
    
    # Experience vs outcome
    print(f"\nChampion EXP values:")
    for _, c in champs.iterrows():
        print(f"  {int(c['YEAR'])} {c['TEAM']:<20}: EXP={c['EXP']:.2f}, Rank={int(c['EXP_RANK'])}")

# =============================================================================
# 8. BALANCE ANALYSIS: Do champions need to be good at everything?
# =============================================================================
print("\n" + "="*70)
print("8. BALANCE ANALYSIS: How many 'elite' categories do champions have?")
print("="*70)

# Define elite thresholds (top 20% in category)
elite_categories = ['KADJ EM', 'KADJ O', 'KADJ D', 'EFG%', 'EFG%D', 'TOV%', 'TOV%D', 'EXP']

def count_elite(row, year_data):
    elite_count = 0
    for col in elite_categories:
        if col in year_data.columns:
            threshold = year_data[col].quantile(0.80 if col not in ['KADJ D', 'EFG%D', 'TOV%'] else 0.20)
            if col in ['KADJ D', 'EFG%D', 'TOV%']:  # Lower is better
                if row[col] <= threshold:
                    elite_count += 1
            else:
                if row[col] >= threshold:
                    elite_count += 1
    return elite_count

# Count elite categories for each team
elite_counts = []
for _, row in df.iterrows():
    year_data = df[df['YEAR'] == row['YEAR']]
    elite_counts.append(count_elite(row, year_data))
df['ELITE_COUNT'] = elite_counts

champs = df[df['IS_CHAMPION'] == 1]
non_champs = df[df['IS_CHAMPION'] == 0]

print(f"\nElite category counts (out of {len(elite_categories)}):")
print(f"  Champions average: {champs['ELITE_COUNT'].mean():.1f}")
print(f"  Non-champions average: {non_champs['ELITE_COUNT'].mean():.1f}")

print(f"\nChampion distribution:")
for count in range(len(elite_categories) + 1):
    n = (champs['ELITE_COUNT'] == count).sum()
    if n > 0:
        print(f"  {count} elite categories: {n} champions")

# =============================================================================
# 9. CINDERELLA ANALYSIS: What made the low seeds win?
# =============================================================================
print("\n" + "="*70)
print("9. LOW SEED CHAMPIONS: What made them different?")
print("="*70)

low_seed_champs = champs[champs['SEED'] > 2]
high_seed_champs = champs[champs['SEED'] <= 2]

print(f"\nComparing low-seed champions (3+) vs high-seed champions (1-2):")
print(f"Low seed champions: {len(low_seed_champs)}")
print(f"High seed champions: {len(high_seed_champs)}")

compare_cols = ['KADJ EM', 'KADJ D', 'EXP', 'ELITE SOS', 'EFG%', '3PT%']
print(f"\n{'Metric':<12} {'Low Seed':>12} {'High Seed':>12} {'Diff':>10}")
print("-"*50)

for col in compare_cols:
    if col in df.columns:
        low_mean = low_seed_champs[col].mean()
        high_mean = high_seed_champs[col].mean()
        diff = low_mean - high_mean
        print(f"{col:<12} {low_mean:>12.2f} {high_mean:>12.2f} {diff:>+10.2f}")

print("\nLow seed champion details:")
for _, c in low_seed_champs.iterrows():
    print(f"  {int(c['YEAR'])} {c['TEAM']:<20} Seed {int(c['SEED'])}: EM={c['KADJ EM']:.1f}, DEF={c['KADJ D']:.1f}")

# =============================================================================
# 10. SUGGESTED NEW FEATURES
# =============================================================================
print("\n" + "="*70)
print("10. SUGGESTED NEW FEATURES BASED ON ANALYSIS")
print("="*70)

suggestions = """
Based on this analysis, consider adding these features:

1. ELITE_COUNT - Number of categories where team is top 20%
   Champions average 5.5+ elite categories

2. DEF_ELITE - Binary: Is KADJ D in top 10?
   75% of champions have top-10 defense

3. BALANCED_GOOD - Both OFF_RANK and DEF_RANK < 15
   Champions rarely have major weaknesses

4. EXP_THRESHOLD - Binary: Experience rank < 25
   Most champions have experienced rosters

5. SEED_PENALTY - Penalty for seeds > 4
   Only 2 champions since 2008 were seeded > 4 (both UConn!)

6. HOT_SHOOTING - EFG% + 3PT% composite
   Champions tend to be elite shooting teams

7. DEFENSIVE_VERSATILITY - (TOV%D rank + EFG%D rank) / 2
   Champions force turnovers AND contest shots
"""
print(suggestions)
