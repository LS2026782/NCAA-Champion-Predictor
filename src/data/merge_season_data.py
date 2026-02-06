"""
Merge and standardize season data from various sources into our unified format.

This script takes manually downloaded data from Barttorvik/KenPom and converts
it to match our existing KenPom Barttorvik.csv format.

Usage:
    python -m src.data.merge_season_data --year 2026
    
Expected input files (in data/raw/):
    - barttorvik_2026.csv (from barttorvik.com)
    - kenpom_2026.csv (optional, from kenpom.com)
    
Output:
    - Appends to data/raw/KenPom Barttorvik.csv
    - Or creates data/raw/KenPom Barttorvik 2026.csv
"""

import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Column mapping from Barttorvik raw export to our format
BARTTORVIK_COLUMN_MAP = {
    'Team': 'TEAM',
    'Conf': 'CONF',
    'Rec': 'RECORD',  # Will parse into W-L
    'AdjOE': 'KADJ O',
    'AdjDE': 'KADJ D', 
    'AdjEM': 'KADJ EM',
    'Barthag': 'BARTHAG',
    'AdjT': 'KADJ T',
    'Luck': 'LUCK',
    'EFG%': 'EFG%',
    'EFGD%': 'EFG%D',
    'TOR': 'TOV%',
    'TORD': 'TOV%D',
    'ORB': 'OREB%',
    'DRB': 'DREB%',
    'FTR': 'FTR',
    'FTRD': 'FTRD',
    '2P%': '2PT%',
    '2P%D': '2PT%D',
    '3P%': '3PT%',
    '3P%D': '3PT%D',
    'Exp': 'EXP',
    'Hgt': 'AVG HGT',
    'Talent': 'TALENT',
    'Elite SOS': 'ELITE SOS',
    'WAB': 'WAB',
}

# Alternative column names (Barttorvik uses different names sometimes)
BARTTORVIK_ALT_NAMES = {
    'Adj OE': 'AdjOE',
    'Adj DE': 'AdjDE', 
    'Adj EM': 'AdjEM',
    'Adj T': 'AdjT',
    'eFG%': 'EFG%',
    'eFG%D': 'EFGD%',
    'TO%': 'TOR',
    'TO%D': 'TORD',
    'Height': 'Hgt',
    'Experience': 'Exp',
}


def load_existing_data() -> pd.DataFrame:
    """Load existing KenPom Barttorvik.csv data."""
    path = PROJECT_ROOT / "data/raw/KenPom Barttorvik.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_barttorvik(year: int) -> pd.DataFrame:
    """Load and standardize Barttorvik CSV export."""
    path = PROJECT_ROOT / f"data/raw/barttorvik_{year}.csv"
    
    if not path.exists():
        print(f"File not found: {path}")
        print(f"Please download from: https://barttorvik.com/trank.php?year={year}")
        return None
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} teams from {path}")
    print(f"Columns found: {df.columns.tolist()}")
    
    # Standardize column names (handle alternative names)
    for alt, standard in BARTTORVIK_ALT_NAMES.items():
        if alt in df.columns:
            df.rename(columns={alt: standard}, inplace=True)
    
    # Map to our format
    renamed = {}
    for src, dst in BARTTORVIK_COLUMN_MAP.items():
        if src in df.columns:
            renamed[src] = dst
    
    df.rename(columns=renamed, inplace=True)
    
    # Add year
    df['YEAR'] = year
    
    # Parse record into W-L if needed
    if 'RECORD' in df.columns:
        df[['W', 'L']] = df['RECORD'].str.split('-', expand=True).astype(int)
        df['WIN%'] = df['W'] / (df['W'] + df['L'])
        df['GAMES'] = df['W'] + df['L']
    
    # Initialize tournament columns (will be filled after Selection Sunday)
    df['SEED'] = None
    df['ROUND'] = None
    
    return df


def validate_required_columns(df: pd.DataFrame) -> bool:
    """Check that all key columns exist."""
    required = ['TEAM', 'YEAR', 'KADJ EM', 'BARTHAG']
    important = ['TALENT', 'EXP', 'AVG HGT', 'EFG%', 'TOV%', 'OREB%']
    
    missing_required = [c for c in required if c not in df.columns]
    missing_important = [c for c in important if c not in df.columns]
    
    if missing_required:
        print(f"ERROR: Missing required columns: {missing_required}")
        return False
    
    if missing_important:
        print(f"WARNING: Missing important columns: {missing_important}")
        print("  Model accuracy may be reduced without these features.")
    
    return True


def merge_with_existing(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new season data with existing historical data."""
    if existing_df is None:
        return new_df
    
    year = new_df['YEAR'].iloc[0]
    
    # Remove any existing data for this year
    existing_df = existing_df[existing_df['YEAR'] != year]
    
    # Ensure column alignment
    all_cols = set(existing_df.columns) | set(new_df.columns)
    for col in all_cols:
        if col not in new_df.columns:
            new_df[col] = None
        if col not in existing_df.columns:
            existing_df[col] = None
    
    # Concatenate
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.sort_values(['YEAR', 'TEAM']).reset_index(drop=True)
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Merge season data into unified format')
    parser.add_argument('--year', type=int, required=True, help='Season year (e.g., 2026)')
    parser.add_argument('--append', action='store_true', help='Append to existing KenPom Barttorvik.csv')
    parser.add_argument('--preview', action='store_true', help='Preview without saving')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"MERGING {args.year} SEASON DATA")
    print(f"{'='*70}\n")
    
    # Load Barttorvik data
    new_df = load_barttorvik(args.year)
    if new_df is None:
        return
    
    # Validate
    if not validate_required_columns(new_df):
        return
    
    print(f"\nData summary for {args.year}:")
    print(f"  Teams: {len(new_df)}")
    print(f"  Columns: {len(new_df.columns)}")
    if 'KADJ EM' in new_df.columns:
        print(f"  Top 5 by KADJ EM:")
        top5 = new_df.nlargest(5, 'KADJ EM')[['TEAM', 'KADJ EM', 'BARTHAG']]
        for _, row in top5.iterrows():
            print(f"    {row['TEAM']}: EM={row['KADJ EM']:.1f}, BARTHAG={row['BARTHAG']:.3f}")
    
    if args.preview:
        print("\nPreview mode - no changes saved.")
        return
    
    if args.append:
        existing = load_existing_data()
        combined = merge_with_existing(new_df, existing)
        output_path = PROJECT_ROOT / "data/raw/KenPom Barttorvik.csv"
        combined.to_csv(output_path, index=False)
        print(f"\nAppended to: {output_path}")
        print(f"Total records: {len(combined)}")
    else:
        output_path = PROJECT_ROOT / f"data/raw/KenPom Barttorvik {args.year}.csv"
        new_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    
    print("\nNext steps:")
    print("  1. After Selection Sunday, add SEED column for tournament teams")
    print("  2. Run predictions: python predict_games.py --year 2026")


if __name__ == '__main__':
    main()
