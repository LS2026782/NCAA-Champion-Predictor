"""
KenPom Historical Data Acquisition Guide and Parser

This module provides tools for acquiring and processing historical KenPom data
for seasons 2002-2007 (the "reconstruction zone").

KenPom Data Overview:
- $25/year subscription at kenpom.com
- Provides adjusted efficiency metrics back to 2002
- Four Factors available with subscription
- Game-by-game logs available (for ELITE SOS calculation)

Required Columns from KenPom:
- Team name
- AdjO (Adjusted Offensive Efficiency)
- AdjD (Adjusted Defensive Efficiency)  
- AdjEM (Adjusted Efficiency Margin)
- AdjT (Adjusted Tempo)
- SOS (Strength of Schedule)
- W, L, Record
- Four Factors (with subscription):
  - eFG%, eFG%D
  - TO%, TO%D
  - ORB%, ORB%D (or OREB%/DREB%)
  - FTR, FTRD
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
KENPOM_DIR = HISTORICAL_DIR / "kenpom"

KENPOM_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COLUMN MAPPING
# =============================================================================

# Map KenPom column names to our standardized names
KENPOM_COLUMN_MAP = {
    # Primary columns
    'team': 'TEAM',
    'conf': 'CONF',
    'w-l': 'RECORD',
    'w': 'W',
    'l': 'L',
    
    # Efficiency metrics
    'adjoe': 'KADJ O',
    'adjo': 'KADJ O',
    'adj o': 'KADJ O',
    'adjde': 'KADJ D',
    'adjd': 'KADJ D',
    'adj d': 'KADJ D',
    'adjem': 'KADJ EM',
    'adj em': 'KADJ EM',
    'em': 'KADJ EM',
    'adjtempo': 'KADJ T',
    'adjt': 'KADJ T',
    'tempo': 'KADJ T',
    'adj t': 'KADJ T',
    
    # SOS
    'sos': 'SOS',
    'sos adjd': 'SOS ADJD',
    'sos adjo': 'SOS ADJO',
    
    # Four Factors - Offense
    'efg%': 'EFG%',
    'efg pct': 'EFG%',
    'to%': 'TOV%',
    'to pct': 'TOV%',
    'or%': 'OREB%',
    'orb%': 'OREB%',
    'ftr': 'FTR',
    'ft rate': 'FTR',
    
    # Four Factors - Defense
    'efg%d': 'EFG%D',
    'opp efg%': 'EFG%D',
    'opp efg pct': 'EFG%D',
    'to%d': 'TOV%D',
    'opp to%': 'TOV%D',
    'dr%': 'DREB%',
    'drb%': 'DREB%',
    'ftrd': 'FTRD',
    'opp ftr': 'FTRD',
    
    # Shooting splits
    '2p%': '2PT%',
    '3p%': '3PT%',
    '2p%d': '2PT%D',
    '3p%d': '3PT%D',
    
    # Rank columns (we'll calculate these)
    'rk': 'RANK',
    'rank': 'RANK',
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to match project conventions.
    
    Args:
        df: Raw KenPom DataFrame
    
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    
    # Normalize existing column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Apply mapping
    rename_map = {}
    for col in df.columns:
        if col in KENPOM_COLUMN_MAP:
            rename_map[col] = KENPOM_COLUMN_MAP[col]
    
    df = df.rename(columns=rename_map)
    
    return df


def parse_kenpom_export(filepath: Path, year: int) -> pd.DataFrame:
    """
    Parse a KenPom data export file.
    
    Handles various export formats:
    - CSV downloads
    - Copy-paste from website (tab-delimited)
    - Excel exports
    
    Args:
        filepath: Path to KenPom data file
        year: Season year (e.g., 2005 for 2004-05 season)
    
    Returns:
        Standardized DataFrame
    """
    logger.info(f"Parsing KenPom data from {filepath}")
    
    # Detect file format
    suffix = filepath.suffix.lower()
    
    if suffix == '.xlsx' or suffix == '.xls':
        df = pd.read_excel(filepath)
    elif suffix == '.csv':
        df = pd.read_csv(filepath)
    else:
        # Try tab-delimited first, then comma
        try:
            df = pd.read_csv(filepath, sep='\t')
            if len(df.columns) < 3:
                df = pd.read_csv(filepath)
        except:
            df = pd.read_csv(filepath)
    
    # Standardize columns
    df = standardize_column_names(df)
    
    # Add year column
    df['YEAR'] = year
    
    # Parse W-L if present
    if 'RECORD' in df.columns:
        wl_split = df['RECORD'].str.extract(r'(\d+)-(\d+)')
        df['W'] = pd.to_numeric(wl_split[0], errors='coerce')
        df['L'] = pd.to_numeric(wl_split[1], errors='coerce')
    
    # Calculate BARTHAG if we have efficiency
    if 'KADJ O' in df.columns and 'KADJ D' in df.columns:
        from reconstruct_historical import calculate_barthag
        df['BARTHAG'] = df.apply(
            lambda row: calculate_barthag(row['KADJ O'], row['KADJ D']),
            axis=1
        )
        logger.info("Calculated BARTHAG from efficiency")
    
    # Log results
    logger.info(f"Parsed {len(df)} teams for {year}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def validate_kenpom_data(df: pd.DataFrame) -> Dict:
    """
    Validate that KenPom data has required columns.
    
    Args:
        df: KenPom DataFrame to validate
    
    Returns:
        Dict with validation results
    """
    required_cols = ['TEAM', 'KADJ O', 'KADJ D', 'KADJ EM']
    optional_cols = ['EFG%', 'TOV%', 'OREB%', 'FTR', 'EFG%D', 'TOV%D', 'DREB%', 'FTRD']
    
    results = {
        'valid': True,
        'missing_required': [],
        'missing_optional': [],
        'team_count': len(df),
        'columns_found': list(df.columns)
    }
    
    for col in required_cols:
        if col not in df.columns:
            results['missing_required'].append(col)
            results['valid'] = False
    
    for col in optional_cols:
        if col not in df.columns:
            results['missing_optional'].append(col)
    
    return results


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features for historical data.
    
    Args:
        df: DataFrame with base KenPom metrics
    
    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()
    
    # BARTHAG (if not already present)
    if 'BARTHAG' not in df.columns and 'KADJ O' in df.columns:
        from reconstruct_historical import calculate_barthag
        df['BARTHAG'] = df.apply(
            lambda row: calculate_barthag(row['KADJ O'], row['KADJ D']),
            axis=1
        )
    
    # BADJ columns (approximate from KenPom if not present)
    if 'BADJ EM' not in df.columns and 'KADJ EM' in df.columns:
        # Barttorvik EM is typically very similar to KenPom
        df['BADJ EM'] = df['KADJ EM'] * 1.0  # Could apply slight adjustment
        df['BADJ O'] = df['KADJ O'] * 1.0
        df['BADJ D'] = df['KADJ D'] * 1.0
    
    if 'BADJ T' not in df.columns and 'KADJ T' in df.columns:
        df['BADJ T'] = df['KADJ T']
    
    # WIN%
    if 'W' in df.columns and 'L' in df.columns:
        df['WIN%'] = df['W'] / (df['W'] + df['L'])
        df['GAMES'] = df['W'] + df['L']
    
    return df


# =============================================================================
# DATA ACQUISITION GUIDE
# =============================================================================

KENPOM_GUIDE = """
================================================================================
                  KENPOM HISTORICAL DATA ACQUISITION GUIDE
================================================================================

OVERVIEW
--------
KenPom ($25/year subscription) provides the base efficiency metrics needed
for 2002-2007 seasons. This data forms the foundation for reconstruction.

REQUIRED DATA
-------------
For each season (2002-2007), you need:
  ✓ Team name
  ✓ AdjO (Adjusted Offensive Efficiency)
  ✓ AdjD (Adjusted Defensive Efficiency)
  ✓ AdjEM (Adjusted Efficiency Margin)
  ✓ AdjT (Adjusted Tempo)
  
OPTIONAL (but helpful):
  • Four Factors (eFG%, TO%, ORB%, FTR) - Offense and Defense
  • SOS (Strength of Schedule)
  • Shooting splits (2PT%, 3PT%)

STEP-BY-STEP INSTRUCTIONS
-------------------------

1. SUBSCRIBE TO KENPOM
   - Go to kenpom.com
   - Subscribe ($25/year)
   - This is essential for historical data access

2. DOWNLOAD HISTORICAL DATA
   For each year (2002-2007):
   
   a) Navigate to that season:
      - Click the year selector at top of page
      - Select the season (e.g., "2004-05")
   
   b) Export the summary table:
      - Right-click on the main table
      - Select all data (Ctrl+A on the table)
      - Copy (Ctrl+C)
      - Paste into Excel or a text file
      
   c) Save as CSV:
      - Save as: data/historical/kenpom/kenpom_YYYY.csv
      - Example: kenpom_2005.csv for 2004-05 season

3. DOWNLOAD FOUR FACTORS (if available)
   - Click "Four Factors" link
   - Export offensive four factors
   - Export defensive four factors
   - Save to same folder

4. PROCESS THE DATA
   After downloading, run:
   
   python src/data/fetch_kenpom_historical.py --parse data/historical/kenpom/kenpom_2005.csv --year 2005

5. VALIDATE
   Run validation to check data quality:
   
   python src/data/fetch_kenpom_historical.py --validate

FILE FORMAT EXAMPLES
--------------------

Your CSV files should look something like:

Team,W-L,AdjEM,AdjO,AdjD,AdjT,Luck,SOS
North Carolina,33-4,+29.45,120.3,90.9,69.8,.035,+8.42
Illinois,37-2,+28.23,119.5,91.3,66.7,.004,+6.84
...

The script will handle column name variations and calculate BARTHAG.

================================================================================
"""


def print_acquisition_guide():
    """Print the KenPom data acquisition guide."""
    print(KENPOM_GUIDE)


def check_historical_data_status() -> Dict:
    """
    Check which years have KenPom historical data.
    
    Returns:
        Dict with status for each year
    """
    status = {}
    
    for year in range(2002, 2008):
        file_patterns = [
            KENPOM_DIR / f"kenpom_{year}.csv",
            KENPOM_DIR / f"kenpom_{year-1}-{str(year)[-2:]}.csv",
            KENPOM_DIR / f"kp_{year}.csv",
        ]
        
        found = None
        for pattern in file_patterns:
            if pattern.exists():
                found = pattern
                break
        
        if found:
            df = pd.read_csv(found)
            status[year] = {
                'status': 'found',
                'file': str(found),
                'teams': len(df),
                'columns': len(df.columns)
            }
        else:
            status[year] = {
                'status': 'missing',
                'file': None,
                'teams': 0
            }
    
    return status


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KenPom Historical Data Tools")
    parser.add_argument('--guide', action='store_true', help='Print acquisition guide')
    parser.add_argument('--status', action='store_true', help='Check data status')
    parser.add_argument('--parse', type=str, help='Parse a KenPom file')
    parser.add_argument('--year', type=int, help='Year for parsing')
    parser.add_argument('--validate', action='store_true', help='Validate all data')
    
    args = parser.parse_args()
    
    if args.guide:
        print_acquisition_guide()
    
    elif args.status:
        print("\n=== KenPom Historical Data Status ===\n")
        status = check_historical_data_status()
        for year, info in sorted(status.items()):
            if info['status'] == 'found':
                print(f"  {year}: ✓ Found - {info['teams']} teams ({info['file']})")
            else:
                print(f"  {year}: ✗ Missing")
        
        # Also check main KenPom Barttorvik file
        main_file = RAW_DIR / "KenPom Barttorvik.csv"
        if main_file.exists():
            df = pd.read_csv(main_file)
            years = sorted(df['YEAR'].unique())
            print(f"\n  Main file: {min(years)}-{max(years)} ({len(df)} team-seasons)")
    
    elif args.parse and args.year:
        filepath = Path(args.parse)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
        else:
            df = parse_kenpom_export(filepath, args.year)
            
            # Validate
            results = validate_kenpom_data(df)
            print(f"\nValidation Results:")
            print(f"  Valid: {results['valid']}")
            print(f"  Teams: {results['team_count']}")
            
            if results['missing_required']:
                print(f"  Missing required: {results['missing_required']}")
            if results['missing_optional']:
                print(f"  Missing optional: {results['missing_optional']}")
            
            # Save processed version
            output = KENPOM_DIR / f"kenpom_{args.year}_processed.csv"
            df.to_csv(output, index=False)
            print(f"\nSaved processed data to: {output}")
    
    elif args.validate:
        print("\n=== Validating All KenPom Data ===\n")
        for file in sorted(KENPOM_DIR.glob("*.csv")):
            df = pd.read_csv(file)
            results = validate_kenpom_data(df)
            
            status = "✓" if results['valid'] else "✗"
            print(f"  {status} {file.name}: {results['team_count']} teams")
            
            if not results['valid']:
                print(f"      Missing: {results['missing_required']}")
    
    else:
        print("""
KenPom Historical Data Tools
============================

Usage:
  python fetch_kenpom_historical.py --guide     # Show acquisition guide
  python fetch_kenpom_historical.py --status    # Check data status
  python fetch_kenpom_historical.py --parse FILE --year YYYY  # Parse file
  python fetch_kenpom_historical.py --validate  # Validate all data

First, run --guide to see how to download data from KenPom.
""")
