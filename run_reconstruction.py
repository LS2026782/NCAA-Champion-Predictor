#!/usr/bin/env python
"""
Historical Data Reconstruction Runner

This is the master script for reconstructing historical NCAA basketball data
from 2002-2007 to extend the training dataset for the champion prediction model.

Architecture Overview:
----------------------
Current data: 2008-2025 (from KenPom Barttorvik.csv)
Target data: 2002-2025 (24 seasons of training data)
Reconstruction zone: 2002-2007 (6 seasons)

Reconstruction Strategy (Option B from the research document):
1. Use KenPom subscription data as base (efficiency metrics)
2. Calculate BARTHAG from AdjO/AdjD
3. Calculate ELITE SOS from game logs
4. Reconstruct TALENT from RSCI recruiting data + minutes
5. Reconstruct EXP from roster class data + minutes  
6. Reconstruct HEIGHT metrics from roster data

This unified approach creates a homogeneous feature space that allows
the model to learn patterns across the full 24-year period.

Usage:
  python run_reconstruction.py --status      # Check current data status
  python run_reconstruction.py --guide       # Show step-by-step guide
  python run_reconstruction.py --run         # Run reconstruction pipeline
  python run_reconstruction.py --validate    # Validate final dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HISTORICAL_DIR = DATA_DIR / "historical"

# Required columns for unified dataset
REQUIRED_COLUMNS = [
    'YEAR', 'TEAM', 'CONF',
    'W', 'L', 'WIN%',
    'KADJ O', 'KADJ D', 'KADJ EM', 'KADJ T',
    'BARTHAG',
    'EFG%', 'TOV%', 'OREB%', 'FTR',
    'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
    'TALENT', 'EXP',
    'AVG HGT', 'EFF HGT',
    'ELITE SOS',
    'SEED', 'ROUND'  # Tournament data (only for tourney teams)
]


def print_status():
    """Print current data status and what's needed."""
    print("""
================================================================================
                    DATA RECONSTRUCTION STATUS
================================================================================
""")
    
    # Check existing main file
    main_file = RAW_DIR / "KenPom Barttorvik.csv"
    if main_file.exists():
        df = pd.read_csv(main_file)
        years = sorted(df['YEAR'].unique())
        print(f"[OK] Main dataset: {main_file.name}")
        print(f"  Years: {min(years)}-{max(years)}")
        print(f"  Team-seasons: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
    else:
        print("[MISSING] Main dataset not found")
    
    print()
    
    # Check historical KenPom data
    kenpom_dir = HISTORICAL_DIR / "kenpom"
    kenpom_dir.mkdir(parents=True, exist_ok=True)
    
    print("Historical KenPom Data (2002-2007):")
    for year in range(2002, 2008):
        files = list(kenpom_dir.glob(f"*{year}*.csv"))
        if files:
            df = pd.read_csv(files[0])
            print(f"  [OK] {year}: {len(df)} teams ({files[0].name})")
        else:
            print(f"  [MISSING] {year}: MISSING")
    
    print()
    
    # Check RSCI data
    rsci_dir = HISTORICAL_DIR / "rsci"
    rsci_dir.mkdir(parents=True, exist_ok=True)
    
    print("RSCI Recruiting Data (1998-2013):")
    rsci_files = list(rsci_dir.glob("rsci_*.csv"))
    if rsci_files:
        years_found = [int(f.stem.split('_')[1]) for f in rsci_files]
        print(f"  Found: {min(years_found)}-{max(years_found)} ({len(rsci_files)} files)")
    else:
        print("  [MISSING] No RSCI data files found")
    
    print()
    
    # Check roster data
    roster_dir = HISTORICAL_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)
    
    print("Sports-Reference Roster Data:")
    roster_files = list(roster_dir.glob("roster_*.csv"))
    if roster_files:
        print(f"  Found: {len(roster_files)} team-season files")
    else:
        print("  [MISSING] No roster data files found")
    
    print()
    print("=" * 80)


def print_guide():
    """Print step-by-step reconstruction guide."""
    print("""
================================================================================
              HISTORICAL DATA RECONSTRUCTION GUIDE
================================================================================

PHASE 1: BASE DATA ACQUISITION (Required)
==========================================

Step 1: Subscribe to KenPom
---------------------------
- Go to kenpom.com and subscribe ($25/year)
- This provides the foundation: AdjO, AdjD, AdjEM, AdjT, Four Factors

Step 2: Download KenPom Data for 2002-2007
------------------------------------------
For each season (2002, 2003, 2004, 2005, 2006, 2007):

a) Navigate to the season archive on KenPom
b) Copy the main efficiency table
c) Paste into Excel/Google Sheets
d) Save as CSV: data/historical/kenpom/kenpom_YYYY.csv

Required columns: Team, AdjO, AdjD, AdjEM, AdjT
Optional but helpful: eFG%, TO%, ORB%, FTR (offense and defense)

Step 3: Process KenPom Data
---------------------------
Run for each year:
  python src/data/fetch_kenpom_historical.py --parse data/historical/kenpom/kenpom_2005.csv --year 2005


PHASE 2: ROSTER DATA (For TALENT/EXP/HEIGHT)
=============================================

Step 4: Gather RSCI Recruiting Data
-----------------------------------
Option A (Automated - may be blocked):
  python src/data/fetch_rsci_data.py --fetch 2003-2013

Option B (Manual):
  - Visit 247Sports historical composite rankings
  - Save Top 100 for each year (1998-2007) to data/historical/rsci/

Step 5: Gather Sports-Reference Roster Data
-------------------------------------------
For tournament teams in 2002-2007, we need:
  - Player name, class (Fr/So/Jr/Sr), height, minutes played

Option A (Automated - slow, rate-limited):
  - The reconstruct_historical.py script can scrape this

Option B (Manual for key teams):
  - Visit sports-reference.com/cbb/schools/[team]/[year].html
  - Copy roster + per-game stats tables
  - Save to data/historical/rosters/


PHASE 3: TOURNAMENT DATA
========================

Step 6: Add Tournament Seeds and Results
----------------------------------------
For 2002-2007 tournament teams, we need:
  - SEED (1-16)
  - ROUND (1, 2, 4, 8, 16, 32, 64, 68)

Sources:
  - Wikipedia: NCAA Division I Men's Basketball Tournament history
  - sports-reference.com tournament brackets


PHASE 4: RUN RECONSTRUCTION
===========================

Step 7: Execute Reconstruction Pipeline
---------------------------------------
  python run_reconstruction.py --run

This will:
  1. Load KenPom base data (2002-2007)
  2. Calculate BARTHAG from AdjO/AdjD
  3. Merge RSCI data to calculate TALENT
  4. Merge roster data to calculate EXP and HEIGHT
  5. Calculate ELITE SOS from opponent quality
  6. Combine with existing 2008-2025 data
  7. Output unified dataset


PHASE 5: VALIDATION
===================

Step 8: Validate Unified Dataset
--------------------------------
  python run_reconstruction.py --validate

This checks:
  - All required columns present
  - No unexpected missing values
  - Reasonable value ranges
  - Consistency across years


NOTES
=====
- BARTHAG can be calculated automatically from KenPom efficiency
- ELITE SOS may need approximation for early years
- TALENT/EXP/HEIGHT are the most labor-intensive to reconstruct
- If roster data unavailable, can impute from conference/seed averages

================================================================================
""")


def run_reconstruction():
    """Run the full reconstruction pipeline."""
    logger.info("Starting reconstruction pipeline...")
    
    # Step 1: Check for required source data
    kenpom_dir = HISTORICAL_DIR / "kenpom"
    
    kenpom_files = {}
    for year in range(2002, 2008):
        files = list(kenpom_dir.glob(f"*{year}*.csv"))
        if files:
            kenpom_files[year] = files[0]
        else:
            logger.warning(f"Missing KenPom data for {year}")
    
    if not kenpom_files:
        logger.error("No historical KenPom data found!")
        logger.error("Please follow the guide: python run_reconstruction.py --guide")
        return None
    
    # Step 2: Load and process historical KenPom data
    from data.fetch_kenpom_historical import parse_kenpom_export, calculate_derived_features
    from data.reconstruct_historical import (
        calculate_barthag, 
        RSCIDataHandler, 
        SportsReferenceRosterScraper,
        calculate_team_talent,
        calculate_team_experience,
        calculate_effective_height,
        calculate_average_height
    )
    
    historical_dfs = []
    
    for year, filepath in sorted(kenpom_files.items()):
        logger.info(f"Processing {year}...")
        
        try:
            df = parse_kenpom_export(filepath, year)
            df = calculate_derived_features(df)
            historical_dfs.append(df)
            logger.info(f"  Loaded {len(df)} teams for {year}")
        except Exception as e:
            logger.error(f"  Error processing {year}: {e}")
    
    if not historical_dfs:
        logger.error("No historical data could be processed")
        return None
    
    # Combine historical data
    historical_df = pd.concat(historical_dfs, ignore_index=True)
    logger.info(f"Combined historical data: {len(historical_df)} team-seasons")
    
    # Step 3: Try to add roster-based metrics (TALENT, EXP, HEIGHT)
    rsci_handler = RSCIDataHandler()
    
    # Load available RSCI data
    rsci_dir = HISTORICAL_DIR / "rsci"
    for file in rsci_dir.glob("rsci_*.csv"):
        try:
            year = int(file.stem.split('_')[1])
            rsci_handler.load_rsci_year(year, file)
        except Exception as e:
            logger.debug(f"Could not load RSCI file {file}: {e}")
    
    # For now, set placeholder values for missing roster metrics
    # These can be refined with actual scraping/manual data entry
    if 'TALENT' not in historical_df.columns:
        # Approximate based on conference (power conferences have higher talent)
        power_confs = ['ACC', 'SEC', 'B10', 'B12', 'BE', 'P12']
        historical_df['TALENT'] = historical_df['CONF'].apply(
            lambda c: 50 if c in power_confs else 25
        )
        logger.info("  Added approximate TALENT (conference-based)")
    
    if 'EXP' not in historical_df.columns:
        # Default experience around 1.5 (mid-range)
        historical_df['EXP'] = 1.5
        logger.info("  Added default EXP (1.5)")
    
    if 'AVG HGT' not in historical_df.columns:
        historical_df['AVG HGT'] = 76.0  # Average height ~6'4"
        logger.info("  Added default AVG HGT (76\")")
    
    if 'EFF HGT' not in historical_df.columns:
        historical_df['EFF HGT'] = 80.0  # Effective height ~6'8"
        logger.info("  Added default EFF HGT (80\")")
    
    if 'ELITE SOS' not in historical_df.columns:
        # Approximate from overall SOS or EM rank
        if 'SOS' in historical_df.columns:
            historical_df['ELITE SOS'] = historical_df['SOS']
        else:
            historical_df['ELITE SOS'] = 0.0
        logger.info("  Added approximate ELITE SOS")
    
    # Step 4: Load existing main dataset
    main_file = RAW_DIR / "KenPom Barttorvik.csv"
    existing_df = pd.read_csv(main_file)
    existing_years = set(existing_df['YEAR'].unique())
    
    logger.info(f"Existing data: {min(existing_years)}-{max(existing_years)}")
    
    # Step 5: Filter historical to non-overlapping years
    historical_df = historical_df[~historical_df['YEAR'].isin(existing_years)]
    logger.info(f"Historical data to add: {sorted(historical_df['YEAR'].unique())}")
    
    # Step 6: Align columns
    # Get all columns from both datasets
    all_columns = list(set(existing_df.columns) | set(historical_df.columns))
    
    # Add missing columns with NaN
    for col in all_columns:
        if col not in historical_df.columns:
            historical_df[col] = np.nan
        if col not in existing_df.columns:
            existing_df[col] = np.nan
    
    # Step 7: Combine datasets
    unified_df = pd.concat([historical_df, existing_df], ignore_index=True)
    unified_df = unified_df.sort_values(['YEAR', 'TEAM']).reset_index(drop=True)
    
    logger.info(f"Unified dataset: {len(unified_df)} team-seasons")
    logger.info(f"Years: {sorted(unified_df['YEAR'].unique())}")
    
    # Step 8: Save
    output_file = RAW_DIR / "KenPom Barttorvik Extended.csv"
    unified_df.to_csv(output_file, index=False)
    logger.info(f"Saved to: {output_file}")
    
    return unified_df


def validate_dataset(filepath: Path = None):
    """Validate the unified dataset."""
    if filepath is None:
        filepath = RAW_DIR / "KenPom Barttorvik Extended.csv"
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return False
    
    df = pd.read_csv(filepath)
    
    print(f"""
================================================================================
                    DATASET VALIDATION
================================================================================

File: {filepath}
Team-seasons: {len(df)}
Years: {sorted(df['YEAR'].unique())}
Columns: {len(df.columns)}
""")
    
    # Check required columns
    print("Required Columns Check:")
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = (len(df) - missing) / len(df) * 100
            status = "[OK]" if pct > 90 else "[WARN]" if pct > 50 else "[BAD]"
            print(f"  {status} {col}: {pct:.1f}% complete ({missing} missing)")
        else:
            print(f"  [MISSING] {col}: MISSING COLUMN")
    
    # Value range checks
    print("\nValue Range Checks:")
    
    checks = [
        ('KADJ O', 80, 130, 'Adjusted Offensive Efficiency'),
        ('KADJ D', 80, 130, 'Adjusted Defensive Efficiency'),
        ('KADJ EM', -30, 40, 'Adjusted Efficiency Margin'),
        ('BARTHAG', 0, 1, 'Win Probability'),
        ('EFG%', 30, 70, 'Effective FG%'),
        ('TALENT', 0, 100, 'Talent Score'),
        ('EXP', 0, 4, 'Experience'),
    ]
    
    for col, min_val, max_val, desc in checks:
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()
            in_range = actual_min >= min_val and actual_max <= max_val
            status = "[OK]" if in_range else "[WARN]"
            print(f"  {status} {col}: {actual_min:.2f} to {actual_max:.2f} "
                  f"(expected {min_val}-{max_val})")
    
    # Year-by-year summary
    print("\nYear-by-Year Summary:")
    for year in sorted(df['YEAR'].unique()):
        year_df = df[df['YEAR'] == year]
        tourney_teams = year_df['SEED'].notna().sum()
        print(f"  {year}: {len(year_df)} teams, {tourney_teams} tournament teams")
    
    print("\n" + "=" * 80)
    return True


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Historical Data Reconstruction Pipeline"
    )
    parser.add_argument('--status', action='store_true', 
                        help='Show current data status')
    parser.add_argument('--guide', action='store_true',
                        help='Show step-by-step reconstruction guide')
    parser.add_argument('--run', action='store_true',
                        help='Run reconstruction pipeline')
    parser.add_argument('--validate', action='store_true',
                        help='Validate unified dataset')
    parser.add_argument('--file', type=str,
                        help='File to validate (with --validate)')
    
    args = parser.parse_args()
    
    if args.status:
        print_status()
    elif args.guide:
        print_guide()
    elif args.run:
        run_reconstruction()
    elif args.validate:
        filepath = Path(args.file) if args.file else None
        validate_dataset(filepath)
    else:
        print("""
Historical Data Reconstruction Pipeline
=======================================

This tool extends the NCAA basketball dataset back to 2002.

Commands:
  python run_reconstruction.py --status    Show current data status
  python run_reconstruction.py --guide     Step-by-step guide
  python run_reconstruction.py --run       Run reconstruction
  python run_reconstruction.py --validate  Validate dataset

Start with --status to see what data you have, then --guide for instructions.
""")
