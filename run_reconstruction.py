#!/usr/bin/env python
"""
Historical Data Reconstruction Runner

Master script for the longitudinal data architecture described in the
"LONGITUDINAL ARCHITECTURE FOR PREDICTIVE MODELING IN COLLEGIATE BASKETBALL"
research document. Extends the training dataset from 2008-2025 to 2002-2025.

Architecture:
  Current data: 2008-2025 (from KenPom Barttorvik.csv / Barttorvik scrape)
  Target data:  2002-2025 (24 seasons, ~22 champions for training)
  Reconstruction zone: 2002-2007 (6 seasons needing feature engineering)

Reconstruction Strategy (Option B — Unified Feature Space):
  Phase 1: KenPom subscription data for base efficiency metrics
  Phase 2: BARTHAG calculated: AdjO^11.5 / (AdjO^11.5 + AdjD^11.5)
  Phase 3: ELITE SOS from opponent game logs (avg AdjEM of Top-50 opponents)
  Phase 4: TALENT from RSCI recruiting data + exponential decay + minutes weighting
  Phase 5: EXP from roster class data (Fr=0, So=1, Jr=2, Sr=3) + minutes weighting
  Phase 6: HEIGHT from roster data (effective height = top 40% of minutes by height)

Why Option B over Option A (Split Models):
  - Preserves all ~22 positive champion samples in a single training set
  - Allows the model to learn Talent-Experience interaction across eras
  - A split model with ~11 positives per window is statistically fragile

Usage:
  python run_reconstruction.py --status      Check current data status
  python run_reconstruction.py --guide       Show step-by-step guide
  python run_reconstruction.py --run         Run reconstruction pipeline
  python run_reconstruction.py --validate    Validate final dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HISTORICAL_DIR = DATA_DIR / "historical"

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
    'SEED', 'ROUND'
]


def print_status():
    """Print current data status and what's needed."""
    from config.settings import MIN_YEAR, MAX_YEAR, COVID_YEAR, RECONSTRUCTION_YEARS
    
    print(f"""
================================================================================
                    DATA RECONSTRUCTION STATUS
================================================================================
  Target range: {MIN_YEAR}-{MAX_YEAR} (excluding {COVID_YEAR})
  Reconstruction zone: {RECONSTRUCTION_YEARS[0]}-{RECONSTRUCTION_YEARS[-1]}
""")
    
    main_file = RAW_DIR / "KenPom Barttorvik.csv"
    if main_file.exists():
        df = pd.read_csv(main_file)
        years = sorted(df['YEAR'].unique())
        print(f"  [OK] Main dataset: {main_file.name}")
        print(f"       Years: {min(years)}-{max(years)} ({len(years)} seasons)")
        print(f"       Team-seasons: {len(df)}")
        
        key_cols = ['BARTHAG', 'TALENT', 'EXP', 'AVG HGT', 'ELITE SOS']
        for col in key_cols:
            if col in df.columns:
                valid = df[col].notna().sum()
                print(f"       {col}: {valid}/{len(df)} ({100*valid/len(df):.0f}%)")
            else:
                print(f"       {col}: MISSING COLUMN")
    else:
        print("  [MISSING] Main dataset not found")
    
    # Check for 2026 data
    file_2026 = RAW_DIR / "KenPom Barttorvik 2026.csv"
    if file_2026.exists():
        df26 = pd.read_csv(file_2026)
        print(f"\n  [OK] 2026 data: {len(df26)} teams")
    else:
        print("\n  [MISSING] 2026 data not found")
    
    # Check extended dataset
    extended = RAW_DIR / "KenPom Barttorvik Extended.csv"
    if extended.exists():
        dfe = pd.read_csv(extended)
        years_ext = sorted(dfe['YEAR'].unique())
        print(f"\n  [OK] Extended dataset: {min(years_ext)}-{max(years_ext)} ({len(dfe)} rows)")
    
    print()
    
    kenpom_dir = HISTORICAL_DIR / "kenpom"
    kenpom_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Historical KenPom Data (2002-2007):")
    for year in RECONSTRUCTION_YEARS:
        files = list(kenpom_dir.glob(f"*{year}*.csv"))
        if files:
            df = pd.read_csv(files[0])
            print(f"    [OK] {year}: {len(df)} teams ({files[0].name})")
        else:
            print(f"    [MISSING] {year}")
    
    print()
    
    rsci_dir = HISTORICAL_DIR / "rsci"
    rsci_dir.mkdir(parents=True, exist_ok=True)
    
    print("  RSCI Recruiting Data (1998-2013):")
    rsci_files = list(rsci_dir.glob("rsci_*.csv"))
    if rsci_files:
        years_found = sorted(int(f.stem.split('_')[1]) for f in rsci_files)
        print(f"    Found: {min(years_found)}-{max(years_found)} ({len(rsci_files)} files)")
    else:
        print("    [MISSING] No RSCI data files found")
    
    print()
    
    roster_dir = HISTORICAL_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Sports-Reference Roster Data:")
    roster_files = list(roster_dir.glob("roster_*.csv"))
    if roster_files:
        print(f"    Found: {len(roster_files)} team-season files")
    else:
        print("    [MISSING] No roster data files found")
    
    print()
    print("=" * 80)


def print_guide():
    """Print step-by-step reconstruction guide."""
    from config.settings import BARTHAG_EXPONENT, TALENT_DECAY_SIGMA, EFF_HEIGHT_MINUTES_FRACTION
    
    print(f"""
================================================================================
              HISTORICAL DATA RECONSTRUCTION GUIDE
================================================================================

The goal is to create a HOMOGENEOUS feature space from 2002-2025. This means
every row in the dataset has the same columns — enabling the model to learn
longitudinal trends (Talent vs Experience across eras, 3-point revolution, etc.)

The reconstruction formulas:

  BARTHAG = AdjO^{BARTHAG_EXPONENT} / (AdjO^{BARTHAG_EXPONENT} + AdjD^{BARTHAG_EXPONENT})
  TALENT  = SUM(100 * e^(-(Rank-1)/{TALENT_DECAY_SIGMA}) * Minutes) / SUM(Minutes)
  EXP     = SUM(ClassValue * Minutes) / SUM(Minutes)
  EFF HGT = Avg height of tallest players covering {EFF_HEIGHT_MINUTES_FRACTION*100:.0f}% of minutes
  ELITE SOS = Avg AdjEM of Top-50 opponents

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
  5. Approximate ELITE SOS
  6. Impute any remaining gaps with era-appropriate defaults
  7. Combine with existing 2008-2025 data
  8. Output unified dataset


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
  - Era distribution (should have ~22 champions)

NOTES
=====
- BARTHAG can be calculated automatically (no external data needed)
- ELITE SOS may need approximation for early years
- TALENT/EXP/HEIGHT are the most labor-intensive to reconstruct
- If roster data is unavailable, era-appropriate defaults are used
- The pipeline is designed to be incremental — run again after adding data

================================================================================
""")


def run_reconstruction():
    """Run the full reconstruction pipeline."""
    from data.reconstruct_historical import (
        create_unified_dataset,
        fill_barthag_where_missing,
        approximate_elite_sos_from_sos,
        impute_missing_roster_metrics,
        RSCIDataHandler,
        SportsReferenceRosterScraper,
        calculate_team_talent,
        calculate_team_experience,
        calculate_effective_height,
        calculate_average_height,
    )
    from config.settings import MIN_YEAR, MAX_YEAR

    logger.info("=" * 70)
    logger.info("HISTORICAL DATA RECONSTRUCTION PIPELINE")
    logger.info(f"Target: {MIN_YEAR}-{MAX_YEAR} unified dataset")
    logger.info("=" * 70)

    # Step 1: Check for required source data
    kenpom_dir = HISTORICAL_DIR / "kenpom"
    kenpom_dir.mkdir(parents=True, exist_ok=True)

    kenpom_files = {}
    for year in range(2002, 2008):
        files = list(kenpom_dir.glob(f"*{year}*.csv"))
        if files:
            kenpom_files[year] = files[0]
        else:
            logger.warning(f"Missing KenPom data for {year}")

    if kenpom_files:
        logger.info(f"Found historical KenPom data for: {sorted(kenpom_files.keys())}")
    else:
        logger.warning("No historical KenPom data found — proceeding with existing data only")

    # Step 2: Load and process any available historical KenPom data
    historical_dfs = []
    if kenpom_files:
        try:
            from data.fetch_kenpom_historical import parse_kenpom_export, calculate_derived_features

            for year, filepath in sorted(kenpom_files.items()):
                logger.info(f"Processing historical {year}...")
                try:
                    df = parse_kenpom_export(filepath, year)
                    df = calculate_derived_features(df)
                    historical_dfs.append(df)
                    logger.info(f"  Loaded {len(df)} teams for {year}")
                except Exception as e:
                    logger.error(f"  Error processing {year}: {e}")
        except ImportError:
            logger.warning("fetch_kenpom_historical module not available — skipping historical parsing")

    # Step 3: Try to add roster-based metrics (TALENT, EXP, HEIGHT)
    rsci_handler = RSCIDataHandler()
    rsci_dir = HISTORICAL_DIR / "rsci"
    rsci_dir.mkdir(parents=True, exist_ok=True)

    for file in rsci_dir.glob("rsci_*.csv"):
        try:
            year = int(file.stem.split('_')[1])
            rsci_handler.load_rsci_year(year, file)
        except Exception as e:
            logger.debug(f"Could not load RSCI file {file}: {e}")

    # Apply era-appropriate defaults for missing roster metrics
    if historical_dfs:
        historical_df = pd.concat(historical_dfs, ignore_index=True)
        logger.info(f"Combined historical data: {len(historical_df)} team-seasons")

        historical_df = fill_barthag_where_missing(historical_df)
        historical_df = approximate_elite_sos_from_sos(historical_df)
        historical_df = impute_missing_roster_metrics(historical_df)
    else:
        historical_df = None

    # Step 4: Create unified dataset
    output_file = RAW_DIR / "KenPom Barttorvik Extended.csv"

    if historical_df is not None:
        # Save historical data temporarily so create_unified_dataset can find it
        temp_hist = HISTORICAL_DIR / "kenpom" / "_combined_historical.csv"
        historical_df.to_csv(temp_hist, index=False)

    unified_df = create_unified_dataset(
        start_year=MIN_YEAR,
        end_year=MAX_YEAR,
        output_file=output_file
    )

    if unified_df is not None and len(unified_df) > 0:
        print(f"\n{'=' * 70}")
        print("RECONSTRUCTION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Output: {output_file}")
        print(f"  Total team-seasons: {len(unified_df)}")
        print(f"  Years: {sorted(unified_df['YEAR'].unique())}")

        tourney_count = unified_df['SEED'].notna().sum()
        champ_count = (unified_df.get('ROUND') == 1).sum() if 'ROUND' in unified_df.columns else 'unknown'
        print(f"  Tournament teams: {tourney_count}")
        print(f"  Champions: {champ_count}")

        print(f"\n  Feature coverage:")
        for col in ['BARTHAG', 'TALENT', 'EXP', 'AVG HGT', 'ELITE SOS']:
            if col in unified_df.columns:
                valid = unified_df[col].notna().sum()
                print(f"    {col:12s}: {valid}/{len(unified_df)} ({100*valid/len(unified_df):.0f}%)")
            else:
                print(f"    {col:12s}: MISSING")
        
        print(f"\n{'=' * 70}")
    else:
        logger.error("Reconstruction produced no data")

    return unified_df


def validate_dataset(filepath: Path = None):
    """Validate the unified dataset."""
    if filepath is None:
        filepath = RAW_DIR / "KenPom Barttorvik Extended.csv"
        if not filepath.exists():
            filepath = RAW_DIR / "KenPom Barttorvik.csv"

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
    all_ok = True
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = (len(df) - missing) / len(df) * 100
            status = "[OK]  " if pct > 90 else "[WARN]" if pct > 50 else "[BAD] "
            if pct < 90:
                all_ok = False
            print(f"  {status} {col}: {pct:.1f}% complete ({missing} missing)")
        else:
            print(f"  [MISS] {col}: MISSING COLUMN")
            all_ok = False

    # Value range checks
    print("\nValue Range Checks:")
    checks = [
        ('KADJ O', 80, 135, 'Adjusted Offensive Efficiency'),
        ('KADJ D', 80, 135, 'Adjusted Defensive Efficiency'),
        ('KADJ EM', -35, 45, 'Adjusted Efficiency Margin'),
        ('BARTHAG', 0, 1, 'Win Probability'),
        ('EFG%', 30, 70, 'Effective FG%'),
        ('TALENT', 0, 100, 'Talent Score'),
        ('EXP', 0, 4, 'Experience'),
        ('AVG HGT', 68, 85, 'Average Height (inches)'),
    ]

    for col, min_val, max_val, desc in checks:
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()
            in_range = actual_min >= min_val * 0.9 and actual_max <= max_val * 1.1
            status = "[OK]  " if in_range else "[WARN]"
            print(f"  {status} {col}: {actual_min:.2f} to {actual_max:.2f} "
                  f"(expected ~{min_val}-{max_val})")

    # Year-by-year summary
    print("\nYear-by-Year Summary:")
    for year in sorted(df['YEAR'].unique()):
        year_df = df[df['YEAR'] == year]
        tourney_teams = year_df['SEED'].notna().sum()
        champ = year_df[year_df.get('ROUND', pd.Series()) == 1]['TEAM'].values
        champ_str = f" | Champion: {champ[0]}" if len(champ) > 0 else ""
        print(f"  {year}: {len(year_df):3d} teams, {tourney_teams:2d} tourney{champ_str}")

    # Era distribution
    from config.settings import ERA_BOUNDARIES
    print("\nEra Distribution:")
    for era_name, (start, end) in ERA_BOUNDARIES.items():
        era_teams = df[(df['YEAR'] >= start) & (df['YEAR'] <= end)]
        era_champs = era_teams[era_teams.get('ROUND', pd.Series()) == 1]
        print(f"  {era_name} ({start}-{end}): {len(era_teams)} team-seasons, "
              f"{len(era_champs)} champions")

    print(f"\n{'=' * 80}")
    return all_ok


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Historical Data Reconstruction Pipeline (2002-2025)"
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
Historical Data Reconstruction Pipeline (2002-2025)
====================================================

Extends NCAA basketball data back to 2002 for longitudinal champion prediction.
Creates a unified feature space with 24 seasons of training data (~22 champions).

Commands:
  python run_reconstruction.py --status    Show current data status
  python run_reconstruction.py --guide     Step-by-step guide
  python run_reconstruction.py --run       Run reconstruction
  python run_reconstruction.py --validate  Validate dataset

Start with --status to see what data you have, then --guide for instructions.

Key formulas:
  BARTHAG = AdjO^11.5 / (AdjO^11.5 + AdjD^11.5)
  TALENT  = weighted_avg(100 * e^(-(Rank-1)/25), Minutes)
  EXP     = weighted_avg(ClassValue, Minutes)
""")
