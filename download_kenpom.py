#!/usr/bin/env python
"""
KenPom Historical Data Downloader

Automated download of ALL historical KenPom data (2002-2025) using kenpompy.
Merges Pomeroy ratings, Four Factors, and Height/Experience into a unified
dataset ready for the reconstruction pipeline.

Prerequisites:
  - KenPom subscription ($25/year at kenpom.com)
  - pip install kenpompy

Usage:
  python download_kenpom.py --email YOUR_EMAIL --password YOUR_PASSWORD

  # Download specific year range:
  python download_kenpom.py --email YOU@EMAIL --password PASS --start 2002 --end 2007

  # Download everything and build unified dataset:
  python download_kenpom.py --email YOU@EMAIL --password PASS --all
"""

import argparse
import time
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HISTORICAL_DIR = DATA_DIR / "historical"
KENPOM_DIR = HISTORICAL_DIR / "kenpom"

for d in [RAW_DIR, KENPOM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DELAY_BETWEEN_REQUESTS = 3  # seconds — be respectful to kenpom.com


def download_season(browser, year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Download all available KenPom data for a single season.

    Fetches three tables and merges them:
      1. Pomeroy Ratings (AdjO, AdjD, AdjEM, AdjT) — available from 1999
      2. Four Factors (eFG%, TO%, ORB%, FTR, defense) — available from 1999
      3. Height/Experience (Ht, Exp, etc.) — available from 2007

    Args:
        browser: Authenticated kenpompy browser
        year: Season year (e.g., 2005 for 2004-05)
        verbose: Print progress

    Returns:
        Merged DataFrame with all available metrics for that season
    """
    import kenpompy.summary as kp
    import kenpompy.misc as kpm

    season_str = str(year)
    dfs = {}

    # 1. Pomeroy Ratings (main table — AdjO, AdjD, AdjEM, AdjT, SOS, etc.)
    if verbose:
        print(f"  [{year}] Fetching Pomeroy Ratings...", end=" ", flush=True)
    try:
        ratings = kpm.get_pomeroy_ratings(browser, season=season_str)
        if ratings is not None and len(ratings) > 0:
            dfs['ratings'] = ratings
            if verbose:
                print(f"{len(ratings)} teams")
        else:
            if verbose:
                print("EMPTY")
    except Exception as e:
        if verbose:
            print(f"FAILED: {e}")
    time.sleep(DELAY_BETWEEN_REQUESTS)

    # 2. Four Factors
    if verbose:
        print(f"  [{year}] Fetching Four Factors...", end=" ", flush=True)
    try:
        ff = kp.get_fourfactors(browser, season=season_str)
        if ff is not None and len(ff) > 0:
            dfs['fourfactors'] = ff
            if verbose:
                print(f"{len(ff)} teams")
        else:
            if verbose:
                print("EMPTY")
    except Exception as e:
        if verbose:
            print(f"FAILED: {e}")
    time.sleep(DELAY_BETWEEN_REQUESTS)

    # 3. Efficiency (tempo details)
    if verbose:
        print(f"  [{year}] Fetching Efficiency...", end=" ", flush=True)
    try:
        eff = kp.get_efficiency(browser, season=season_str)
        if eff is not None and len(eff) > 0:
            dfs['efficiency'] = eff
            if verbose:
                print(f"{len(eff)} teams")
        else:
            if verbose:
                print("EMPTY")
    except Exception as e:
        if verbose:
            print(f"FAILED: {e}")
    time.sleep(DELAY_BETWEEN_REQUESTS)

    # 4. Height/Experience (only available from 2007+)
    if year >= 2007:
        if verbose:
            print(f"  [{year}] Fetching Height/Experience...", end=" ", flush=True)
        try:
            ht = kp.get_height(browser, season=season_str)
            if ht is not None and len(ht) > 0:
                dfs['height'] = ht
                if verbose:
                    print(f"{len(ht)} teams")
            else:
                if verbose:
                    print("EMPTY")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
        time.sleep(DELAY_BETWEEN_REQUESTS)

    if not dfs:
        print(f"  [{year}] WARNING: No data fetched!")
        return pd.DataFrame()

    # Merge all tables on Team name
    merged = None
    for name, df in dfs.items():
        # Normalize team column
        team_col = None
        for candidate in ['Team', 'team', 'TEAM', 'TeamName']:
            if candidate in df.columns:
                team_col = candidate
                break

        if team_col is None:
            if verbose:
                print(f"  [{year}] WARNING: No team column found in {name} table")
                print(f"         Columns: {list(df.columns)[:10]}")
            continue

        df = df.rename(columns={team_col: 'Team'})
        # Clean team names (remove seed numbers that KenPom sometimes appends)
        df['Team'] = df['Team'].astype(str).str.strip()
        df['Team'] = df['Team'].str.replace(r'\s*\d+$', '', regex=True)

        if merged is None:
            merged = df
        else:
            # Drop overlapping columns before merge (except Team)
            overlap = set(merged.columns) & set(df.columns) - {'Team'}
            if overlap:
                df = df.drop(columns=list(overlap), errors='ignore')
            merged = merged.merge(df, on='Team', how='outer')

    if merged is None or len(merged) == 0:
        return pd.DataFrame()

    merged['YEAR'] = year

    if verbose:
        print(f"  [{year}] Merged: {len(merged)} teams, {len(merged.columns)} columns")

    return merged


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map kenpompy column names to our project's standardized names.

    kenpompy returns columns like 'off-efg%', 'def-to%', 'avghgt', 'effhgt',
    'sos-adjem', 'seed', etc. This maps them to our standard format.
    """
    rename_map = {
        # Identity (already standardized by download_season)
        'Team': 'TEAM',

        # Conference
        'conference': 'CONF',

        # Record
        'w-l': 'RECORD',

        # Efficiency — kenpompy uses both styles depending on table
        'adjem': 'KADJ EM',
        'adjoe': 'KADJ O',
        'adjde': 'KADJ D',
        'adjtempo': 'KADJ T',
        'adjo': 'KADJ O',
        'adjd': 'KADJ D',
        'adjt': 'KADJ T',

        # Four Factors — Offense (kenpompy format: 'off-efg%')
        'off-efg%': 'EFG%',
        'off-to%': 'TOV%',
        'off-or%': 'OREB%',
        'off-ftrate': 'FTR',

        # Four Factors — Defense (kenpompy format: 'def-efg%')
        'def-efg%': 'EFG%D',
        'def-to%': 'TOV%D',
        'def-or%': 'DREB%',
        'def-ftrate': 'FTRD',

        # Height/Experience (kenpompy format: 'avghgt', 'effhgt')
        'avghgt': 'AVG HGT',
        'effhgt': 'EFF HGT',
        'experience': 'EXP',
        'continuity': 'CONTINUITY',

        # SOS
        'sos-adjem': 'SOS',
        'ncsos-adjem': 'NCSOS',

        # Seed (kenpompy provides this!)
        'seed': 'SEED',

        # Rank
        'rk': 'RANK',

        # Luck
        'luck': 'LUCK',
    }

    existing_targets = set(df.columns)
    final_rename = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        target = rename_map.get(col_lower) or rename_map.get(col)
        if target and target != col and target not in existing_targets:
            final_rename[col] = target
            existing_targets.add(target)

    df = df.rename(columns=final_rename)

    # Drop rank columns (we don't need them as features)
    rank_cols = [c for c in df.columns if c.endswith('.rank')]
    df = df.drop(columns=rank_cols, errors='ignore')

    # Drop duplicate/raw columns that we've already mapped to standard names
    raw_dupes = ['off. efficiency-adj', 'def. efficiency-adj',
                 'off. efficiency-raw', 'def. efficiency-raw',
                 'tempo-adj', 'tempo-raw']
    df = df.drop(columns=[c for c in raw_dupes if c in df.columns], errors='ignore')

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate BARTHAG and other derived features."""
    from src.data.reconstruct_historical import calculate_barthag

    df = df.copy()

    # Ensure efficiency columns are numeric
    for col in ['KADJ O', 'KADJ D', 'KADJ EM', 'KADJ T']:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # BARTHAG
    if 'KADJ O' in df.columns and 'KADJ D' in df.columns:
        if 'BARTHAG' not in df.columns or df['BARTHAG'].isna().all():
            df['BARTHAG'] = df.apply(
                lambda r: calculate_barthag(r['KADJ O'], r['KADJ D']), axis=1
            )

    # KADJ EM (if not already present)
    if 'KADJ EM' not in df.columns and 'KADJ O' in df.columns:
        df['KADJ EM'] = df['KADJ O'] - df['KADJ D']

    # WIN%
    if 'RECORD' in df.columns and 'WIN%' not in df.columns:
        parts = df['RECORD'].astype(str).str.extract(r'(\d+)-(\d+)')
        df['W'] = pd.to_numeric(parts[0], errors='coerce')
        df['L'] = pd.to_numeric(parts[1], errors='coerce')
        df['WIN%'] = df['W'] / (df['W'] + df['L'])
        df['GAMES'] = df['W'] + df['L']

    return df


def download_all(email: str, password: str,
                 start_year: int = 2002, end_year: int = 2025,
                 save_individual: bool = True) -> pd.DataFrame:
    """
    Download all seasons and build a unified dataset.

    Args:
        email: KenPom login email
        password: KenPom login password
        start_year: First season to download (default 2002)
        end_year: Last season to download (default 2025)
        save_individual: Whether to save per-year CSVs

    Returns:
        Unified DataFrame with all seasons
    """
    from kenpompy.utils import login

    print("=" * 70)
    print(f"KENPOM HISTORICAL DATA DOWNLOAD ({start_year}-{end_year})")
    print("=" * 70)

    print("\nLogging into KenPom...", end=" ", flush=True)
    try:
        browser = login(email, password)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")
        print("\nMake sure your email and password are correct.")
        print("Subscribe at https://kenpom.com if you haven't already.")
        return pd.DataFrame()

    all_seasons = []
    skip_years = {2020}  # No tournament

    for year in range(start_year, end_year + 1):
        if year in skip_years:
            print(f"\n[{year}] Skipping (no tournament)")
            continue

        print(f"\n{'-' * 50}")
        print(f"Downloading {year-1}-{str(year)[-2:]} season...")
        print(f"{'-' * 50}")

        df = download_season(browser, year)

        if df is not None and len(df) > 0:
            df = standardize_columns(df)
            df = add_derived_features(df)

            if save_individual:
                out_path = KENPOM_DIR / f"kenpom_{year}.csv"
                df.to_csv(out_path, index=False)
                print(f"  Saved: {out_path.name} ({len(df)} teams, {len(df.columns)} cols)")

            all_seasons.append(df)
        else:
            print(f"  WARNING: No data for {year}")

    if not all_seasons:
        print("\nERROR: No data downloaded!")
        return pd.DataFrame()

    # Build unified dataset
    print(f"\n{'=' * 70}")
    print("BUILDING UNIFIED DATASET")
    print(f"{'=' * 70}")

    unified = pd.concat(all_seasons, ignore_index=True)

    # Convert numeric columns
    numeric_cols = ['KADJ O', 'KADJ D', 'KADJ EM', 'KADJ T', 'BARTHAG',
                    'EFG%', 'TOV%', 'OREB%', 'FTR',
                    'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
                    'EXP', 'AVG HGT', 'EFF HGT',
                    'W', 'L', 'WIN%', 'GAMES',
                    '2PT%', '3PT%', '2PT%D', '3PT%D']
    for col in numeric_cols:
        if col in unified.columns:
            unified[col] = pd.to_numeric(unified[col], errors='coerce')

    unified = unified.sort_values(['YEAR', 'TEAM']).reset_index(drop=True)

    # Placeholder columns for tournament data (to be filled later)
    if 'SEED' not in unified.columns:
        unified['SEED'] = np.nan
    if 'ROUND' not in unified.columns:
        unified['ROUND'] = np.nan

    # Save unified dataset
    output_file = RAW_DIR / "KenPom Barttorvik.csv"
    unified.to_csv(output_file, index=False)

    print(f"\nSaved unified dataset: {output_file}")
    print(f"  Total: {len(unified)} team-seasons")
    print(f"  Years: {sorted(unified['YEAR'].unique())}")
    print(f"  Columns: {len(unified.columns)}")

    # Report per-year summary
    print(f"\nPer-Year Summary:")
    for year in sorted(unified['YEAR'].unique()):
        yr = unified[unified['YEAR'] == year]
        has_barthag = yr['BARTHAG'].notna().sum() if 'BARTHAG' in yr.columns else 0
        has_exp = yr['EXP'].notna().sum() if 'EXP' in yr.columns else 0
        print(f"  {int(year)}: {len(yr):3d} teams | BARTHAG: {has_barthag} | EXP: {has_exp}")

    # Report column coverage
    print(f"\nColumn Coverage:")
    key_cols = ['KADJ O', 'KADJ D', 'KADJ EM', 'BARTHAG',
                'EFG%', 'TOV%', 'OREB%', 'FTR',
                'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
                'EXP', 'AVG HGT', 'EFF HGT']
    for col in key_cols:
        if col in unified.columns:
            valid = unified[col].notna().sum()
            pct = 100 * valid / len(unified)
            print(f"  {col:12s}: {valid:5d}/{len(unified)} ({pct:.0f}%)")
        else:
            print(f"  {col:12s}: MISSING")

    print(f"\n{'=' * 70}")
    print("DONE — Next steps:")
    print("  1. Add tournament SEED and ROUND data for 2002-2007")
    print("  2. Run: python run_reconstruction.py --validate")
    print("  3. Run: python run_reconstruction.py --run")
    print(f"{'=' * 70}")

    return unified


def reprocess_existing():
    """
    Re-process already-downloaded per-year CSVs with corrected column mapping.

    Reads from data/historical/kenpom/kenpom_YYYY.csv, re-standardizes columns,
    and rebuilds the unified dataset.
    """
    print("=" * 70)
    print("RE-PROCESSING EXISTING KENPOM DATA")
    print("=" * 70)

    all_seasons = []

    for csv_file in sorted(KENPOM_DIR.glob("kenpom_*.csv")):
        if '_processed' in csv_file.name or '_combined' in csv_file.name:
            continue
        try:
            year_str = csv_file.stem.split('_')[1]
            year = int(year_str)
        except (IndexError, ValueError):
            continue

        df = pd.read_csv(csv_file)
        df = standardize_columns(df)
        df = add_derived_features(df)

        if 'YEAR' not in df.columns:
            df['YEAR'] = year

        print(f"  {year}: {len(df)} teams, {len(df.columns)} cols "
              f"| EFG%={'YES' if 'EFG%' in df.columns else 'NO'} "
              f"| EXP={'YES' if 'EXP' in df.columns and df['EXP'].notna().any() else 'NO'} "
              f"| HGT={'YES' if 'AVG HGT' in df.columns and df['AVG HGT'].notna().any() else 'NO'}")

        all_seasons.append(df)

    if not all_seasons:
        print("No data files found!")
        return

    unified = pd.concat(all_seasons, ignore_index=True)

    # Convert numeric columns
    numeric_cols = ['KADJ O', 'KADJ D', 'KADJ EM', 'KADJ T', 'BARTHAG',
                    'EFG%', 'TOV%', 'OREB%', 'FTR',
                    'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
                    'EXP', 'AVG HGT', 'EFF HGT',
                    'W', 'L', 'WIN%', 'GAMES', 'SOS', 'NCSOS',
                    '2PT%', '3PT%', '2PT%D', '3PT%D']
    for col in numeric_cols:
        if col in unified.columns:
            unified[col] = pd.to_numeric(unified[col], errors='coerce')

    unified = unified.sort_values(['YEAR', 'TEAM']).reset_index(drop=True)

    if 'SEED' not in unified.columns:
        unified['SEED'] = np.nan
    if 'ROUND' not in unified.columns:
        unified['ROUND'] = np.nan

    output_file = RAW_DIR / "KenPom Barttorvik.csv"
    unified.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"  Total: {len(unified)} team-seasons")
    years = sorted(unified['YEAR'].unique())
    print(f"  Years: {int(min(years))}-{int(max(years))} ({len(years)} seasons)")

    print(f"\nColumn Coverage:")
    key_cols = ['KADJ O', 'KADJ D', 'KADJ EM', 'BARTHAG',
                'EFG%', 'TOV%', 'OREB%', 'FTR',
                'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
                'EXP', 'AVG HGT', 'EFF HGT', 'SOS', 'SEED']
    for col in key_cols:
        if col in unified.columns:
            valid = unified[col].notna().sum()
            pct = 100 * valid / len(unified)
            print(f"  {col:12s}: {valid:5d}/{len(unified)} ({pct:.0f}%)")
        else:
            print(f"  {col:12s}: MISSING")

    print(f"\n{'=' * 70}")
    return unified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download historical KenPom data using your subscription"
    )
    parser.add_argument('--email', type=str,
                        help='KenPom login email')
    parser.add_argument('--password', type=str,
                        help='KenPom login password')
    parser.add_argument('--start', type=int, default=2002,
                        help='First season to download (default: 2002)')
    parser.add_argument('--end', type=int, default=2025,
                        help='Last season to download (default: 2025)')
    parser.add_argument('--all', action='store_true',
                        help='Download all seasons 2002-2025')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-process existing downloads with corrected column mapping')

    args = parser.parse_args()

    if args.reprocess:
        reprocess_existing()
    elif args.email and args.password:
        start = args.start
        end = args.end
        if args.all:
            start = 2002
            end = 2025

        download_all(
            email=args.email,
            password=args.password,
            start_year=start,
            end_year=end
        )
    else:
        print("Usage:")
        print("  Download:    python download_kenpom.py --email X --password Y --all")
        print("  Reprocess:   python download_kenpom.py --reprocess")
