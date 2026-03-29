"""
Build the final pipeline-ready KenPom Barttorvik 2026.csv.

1. Reads fresh Barttorvik data (just scraped)
2. Renames columns to pipeline format
3. Merges ELITE SOS from the March-2 backup
4. Merges TALENT / EXP / EFF HGT from Extended CSV (2025 as proxy)
5. Adds real 2026 NCAA Tournament seeds (Selection Sunday, March 15 2026)
6. Saves the updated file
"""
import pandas as pd
import numpy as np

# ── 2026 NCAA TOURNAMENT BRACKET (Selection Sunday, March 15 2026) ──
# Team names must match Barttorvik format in the CSV
BRACKET_2026_SEEDS = {
    # 1-seeds
    'Duke': 1, 'Arizona': 1, 'Michigan': 1, 'Florida': 1,
    # 2-seeds
    'Houston': 2, 'Connecticut': 2, 'Iowa St.': 2, 'Purdue': 2,
    # 3-seeds
    'Michigan St.': 3, 'Illinois': 3, 'Gonzaga': 3, 'Virginia': 3,
    # 4-seeds
    'Nebraska': 4, 'Alabama': 4, 'Kansas': 4, 'Arkansas': 4,
    # 5-seeds
    'Vanderbilt': 5, "St. John's": 5, 'Texas Tech': 5, 'Wisconsin': 5,
    # 6-seeds
    'Tennessee': 6, 'North Carolina': 6, 'Louisville': 6, 'BYU': 6,
    # 7-seeds
    'Kentucky': 7, "Saint Mary's": 7, 'Miami FL': 7, 'UCLA': 7,
    # 8-seeds
    'Clemson': 8, 'Villanova': 8, 'Ohio St.': 8, 'Georgia': 8,
    # 9-seeds
    'Utah St.': 9, 'TCU': 9, 'Saint Louis': 9, 'Iowa': 9,
    # 10-seeds
    'Santa Clara': 10, 'UCF': 10, 'Missouri': 10, 'Texas A&M': 10,
    # 11-seeds (includes First Four)
    'N.C. State': 11, 'Texas': 11, 'SMU': 11, 'Miami OH': 11,
    'VCU': 11, 'South Florida': 11,
    # 12-seeds
    'McNeese St.': 12, 'Akron': 12, 'High Point': 12, 'Northern Iowa': 12,
    # 13-seeds
    'Cal Baptist': 13, 'Hofstra': 13, 'Troy': 13, 'Hawaii': 13,
    # 14-seeds
    'North Dakota St.': 14, 'Penn': 14, 'Wright St.': 14, 'Kennesaw St.': 14,
    # 15-seeds
    'Idaho': 15, 'Furman': 15, 'Queens': 15, 'Tennessee St.': 15,
    # 16-seeds (includes First Four)
    'Siena': 16, 'LIU': 16, 'Howard': 16, 'UMBC': 16,
    'Prairie View A&M': 16, 'Lehigh': 16,
}

# ── Column mapping: Barttorvik raw → pipeline format ──
COLUMN_MAP = {
    'Team': 'TEAM',
    'Conf': 'CONF',
    'Rec': 'RECORD',
    'AdjOE': 'KADJ O',
    'AdjDE': 'KADJ D',
    'AdjEM': 'KADJ EM',
    'Barthag': 'BARTHAG',
    'AdjT': 'KADJ T',
    'EFGD%': 'EFG%D',
    'TOR': 'TOV%',
    'TORD': 'TOV%D',
    'ORB': 'OREB%',
    'DRB': 'DREB%',
    '2P%': '2PT%',
    '2P%D': '2PT%D',
    '3P%': '3PT%',
    '3P%D': '3PT%D',
    'Elite SOS': 'ELITE SOS',
}

print("=" * 70)
print("BUILDING PIPELINE-READY 2026 DATA")
print("=" * 70)

# ── Step 1: Load fresh Barttorvik data ──
fresh = pd.read_csv('data/raw/barttorvik_2026.csv')
print(f"\n[1/5] Fresh Barttorvik data: {len(fresh)} teams")

# ── Step 2: Rename columns ──
df = fresh.rename(columns=COLUMN_MAP)
df['WIN%'] = df['W'] / df['G']
df['GAMES'] = df['G']
print(f"[2/5] Columns renamed to pipeline format")

# ── Step 3: Merge ELITE SOS from March 2 backup ──
old = pd.read_csv('data/raw/KenPom Barttorvik 2026.csv')
sos_map = old.set_index('TEAM')['ELITE SOS'].to_dict()
df['ELITE SOS'] = df['TEAM'].map(sos_map)
sos_filled = df['ELITE SOS'].notna().sum()
print(f"[3/5] ELITE SOS merged from March 2 data: {sos_filled}/{len(df)} teams")

# ── Step 4: Merge TALENT / EXP / EFF HGT from Extended CSV (2025 proxy) ──
ext = pd.read_csv('data/raw/KenPom Barttorvik Extended.csv')
ext_2025 = ext[ext['YEAR'] == 2025][['TEAM', 'TALENT', 'EXP', 'EFF HGT']].copy()

df = df.merge(ext_2025, on='TEAM', how='left')
talent_filled = df['TALENT'].notna().sum()
exp_filled = df['EXP'].notna().sum()
print(f"[4/5] TALENT merged: {talent_filled}/{len(df)}, EXP merged: {exp_filled}/{len(df)}")

# Fill missing TALENT/EXP with 2025 medians for teams not in 2025 data
for col in ['TALENT', 'EXP', 'EFF HGT']:
    median_val = ext_2025[col].median()
    missing_before = df[col].isna().sum()
    df[col] = df[col].fillna(median_val)
    if missing_before > 0:
        print(f"     {col}: filled {missing_before} missing with median ({median_val:.1f})")

# ── Step 5: Add tournament seeds ──
df['SEED'] = df['TEAM'].map(BRACKET_2026_SEEDS)
df['ROUND'] = np.nan  # Unknown (tournament hasn't happened)

seeded_count = df['SEED'].notna().sum()
print(f"[5/5] Tournament seeds applied: {seeded_count} teams")

# Show any seed teams that didn't match
bracket_teams = set(BRACKET_2026_SEEDS.keys())
data_teams = set(df['TEAM'].values)
unmatched = bracket_teams - data_teams
if unmatched:
    print(f"\n  WARNING: {len(unmatched)} bracket teams not found in data:")
    for t in sorted(unmatched):
        print(f"    - {t} (seed {BRACKET_2026_SEEDS[t]})")

# ── Final column order ──
final_columns = [
    'TEAM', 'CONF', 'G', 'RECORD', 'W', 'L',
    'KADJ O', 'KADJ D', 'KADJ EM', 'BARTHAG', 'KADJ T',
    'EFG%', 'EFG%D', 'TOV%', 'TOV%D', 'OREB%', 'DREB%',
    'FTR', 'FTRD', '2PT%', '2PT%D', '3PT%', '3PT%D', '3PR', '3PRD',
    'ELITE SOS', 'WAB', 'AVG HGT', 'EFF HGT',
    'TALENT', 'EXP',
    'YEAR', 'WIN%', 'GAMES', 'SEED', 'ROUND',
]
df = df[[c for c in final_columns if c in df.columns]]
df = df.sort_values('KADJ EM', ascending=False).reset_index(drop=True)

# ── Save ──
output_path = 'data/raw/KenPom Barttorvik 2026.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} teams to {output_path}")

# ── Display tournament field ──
tourney = df[df['SEED'].notna()].copy()
tourney['SEED'] = tourney['SEED'].astype(int)

print(f"\n{'=' * 70}")
print("2026 NCAA TOURNAMENT FIELD (Official)")
print(f"{'=' * 70}")
for seed in range(1, 17):
    teams = tourney[tourney['SEED'] == seed].sort_values('KADJ EM', ascending=False)
    names = [f"{r['TEAM']} ({r['KADJ EM']:+.1f})" for _, r in teams.iterrows()]
    print(f"  {seed:2d}-seeds: {', '.join(names)}")

print(f"\n{'=' * 70}")
print("TOP 15 TOURNAMENT TEAMS BY EFFICIENCY")
print(f"{'=' * 70}")
for i, (_, r) in enumerate(tourney.sort_values('KADJ EM', ascending=False).head(15).iterrows(), 1):
    print(f"  {i:2d}. ({int(r['SEED'])}) {r['TEAM']:20s} | EM: {r['KADJ EM']:+6.1f} | "
          f"BARTHAG: {r['BARTHAG']:.4f} | SOS: {r['ELITE SOS']:.0f} | "
          f"TAL: {r['TALENT']:.0f} | EXP: {r['EXP']:.2f}")

print(f"\nTotal teams: {len(df)}")
print(f"Tournament teams: {seeded_count}")
print(f"Data columns: {len(df.columns)}")
