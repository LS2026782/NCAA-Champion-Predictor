"""
Convert downloaded 2026 Barttorvik data to our standard format.
Uses 2025 data for TALENT/EXP/HEIGHT as approximations.
"""
import pandas as pd
import numpy as np

print("="*70)
print("CONVERTING 2026 BARTTORVIK DATA")
print("="*70)

# Load downloaded 2026 data
df_2026 = pd.read_csv('data/raw/barttorvik_2026_raw.csv')
print(f"\nLoaded {len(df_2026)} teams from 2026 data")

# Load 2025 data for TALENT/EXP/HEIGHT and four-factors approximations
df_all = pd.read_csv('data/raw/KenPom Barttorvik.csv')
cols_to_copy = ['TEAM', 'TALENT', 'EXP', 'AVG HGT', 'EFF HGT', 
                'EFG%', 'EFG%D', 'TOV%', 'TOV%D', 'OREB%', 'DREB%', 'FTR', 'FTRD',
                '2PT%', '2PT%D', '3PT%', '3PT%D', 'BLK%', 'AST%']
cols_available = [c for c in cols_to_copy if c in df_all.columns]
df_2025 = df_all[df_all['YEAR'] == 2025][cols_available].copy()
print(f"Loaded {len(df_2025)} teams from 2025 for TALENT/EXP/four-factors")

# Column mapping from Barttorvik raw -> our format
column_map = {
    'team': 'TEAM',
    'conf': 'CONF',
    'record': 'RECORD',
    'adjoe': 'KADJ O',
    'adjde': 'KADJ D',
    'barthag': 'BARTHAG',
    'adjt': 'KADJ T',
    'elite SOS': 'ELITE SOS',
    'WAB': 'WAB',
    'sos': 'SOS',
}

# Rename columns
df = df_2026.rename(columns=column_map)

# Calculate KADJ EM (efficiency margin = offense - defense)
df['KADJ EM'] = df['KADJ O'] - df['KADJ D']

# Parse record into W-L
df[['W', 'L']] = df['RECORD'].str.split('-', expand=True).astype(int)
df['GAMES'] = df['W'] + df['L']
df['WIN%'] = df['W'] / df['GAMES']

# Add year
df['YEAR'] = 2026

# Merge TALENT/EXP/HEIGHT from 2025 (as approximations)
df = df.merge(df_2025, on='TEAM', how='left')

# Fill missing values with averages
missing_talent = df['TALENT'].isna().sum()
for col in cols_available:
    if col != 'TEAM' and col in df.columns:
        avg = df[col].mean()
        if pd.isna(avg):
            avg = df_all[df_all['YEAR'] == 2025][col].mean() if col in df_all.columns else 0
        df[col] = df[col].fillna(avg)

print(f"\nMetrics matched from 2025: {len(df) - missing_talent} teams")
print(f"Metrics filled with averages: {missing_talent} teams")

# Add derived features
df['SEED_STRENGTH'] = 0  # Will be updated when seeds are added
df['SEED'] = None  # Tournament seed (added after Selection Sunday)
df['SEED_NUM'] = 16  # Default

# Scale ELITE SOS to match historical range (0-100 instead of 0-1)
df['ELITE SOS'] = df['ELITE SOS'] * 100

# Select and order final columns
final_columns = [
    'YEAR', 'TEAM', 'CONF', 'W', 'L', 'WIN%', 'GAMES',
    'KADJ EM', 'KADJ O', 'KADJ D', 'KADJ T', 'BARTHAG',
    'EFG%', 'EFG%D', 'TOV%', 'TOV%D', 'OREB%', 'DREB%', 'FTR', 'FTRD',
    '2PT%', '2PT%D', '3PT%', '3PT%D', 'BLK%', 'AST%',
    'TALENT', 'EXP', 'AVG HGT', 'EFF HGT',
    'ELITE SOS', 'WAB', 'SOS',
    'SEED', 'SEED_NUM', 'SEED_STRENGTH'
]

# Keep only columns that exist
df_final = df[[c for c in final_columns if c in df.columns]].copy()

# Sort by KADJ EM
df_final = df_final.sort_values('KADJ EM', ascending=False).reset_index(drop=True)

# Save
output_path = 'data/raw/KenPom Barttorvik 2026.csv'
df_final.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print("2026 DATA SUMMARY")
print(f"{'='*70}")
print(f"\nTotal teams: {len(df_final)}")
print(f"Columns: {len(df_final.columns)}")
print(f"\nTop 15 by Adjusted Efficiency Margin:")
print("-"*70)
for i, row in df_final.head(15).iterrows():
    print(f"{i+1:2}. {row['TEAM']:22} ({row['CONF']:4}) | EM: {row['KADJ EM']:+6.2f} | "
          f"BARTHAG: {row['BARTHAG']:.4f} | TALENT: {row['TALENT']:.1f}")

print(f"\n{'='*70}")
print(f"Saved to: {output_path}")
print(f"{'='*70}")

print("\nNOTE: TALENT/EXP/HEIGHT are approximated from 2025 data.")
print("For most accurate predictions, manually update TALENT after")
print("reviewing current rosters at barttorvik.com/hgt_exp.php")

print("\nNext steps:")
print("  1. After Selection Sunday, run: python add_tournament_seeds.py --year 2026")
print("  2. Then run predictions: python predict_games.py --year 2026")
