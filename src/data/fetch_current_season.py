"""
Fetch current season (2025-26) data from KenPom and Barttorvik.

DATA SOURCES:
=============
1. KenPom (kenpom.com) - Requires subscription ($24.95/year)
   - Install: pip install kenpompy
   - Best for: KADJ EM, KADJ O, KADJ D, tempo, etc.

2. Barttorvik (barttorvik.com) - FREE
   - Direct CSV export available
   - Best for: BARTHAG, TALENT, EXP, height, shooting splits

USAGE:
======
# Fetch from Barttorvik (free):
python -m src.data.fetch_current_season --source barttorvik --year 2026

# Fetch from KenPom (requires credentials):
python -m src.data.fetch_current_season --source kenpom --year 2026 --email YOUR_EMAIL --password YOUR_PASSWORD

# Combine both sources (recommended):
python -m src.data.fetch_current_season --source both --year 2026 --email YOUR_EMAIL --password YOUR_PASSWORD
"""

import argparse
import pandas as pd
import requests
from pathlib import Path
from io import StringIO
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def fetch_barttorvik(year: int = 2026) -> pd.DataFrame:
    """
    Fetch team stats from Barttorvik.
    
    The URL pattern for CSV export:
    https://barttorvik.com/trank.php?year=YYYY&csv=1
    
    Note: May require manual download if browser verification blocks automated access.
    """
    print(f"Fetching Barttorvik data for {year} season...")
    
    # Direct CSV URL 
    # Note: The site sometimes requires browser verification
    # If this fails, download manually from the website
    url = f"https://barttorvik.com/team-tables_each.php?year={year}&csv=1"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200 and 'team' in response.text.lower():
            df = pd.read_csv(StringIO(response.text))
            print(f"  Successfully fetched {len(df)} teams from Barttorvik")
            return df
        else:
            print("  Barttorvik requires browser verification. Please download manually:")
            print(f"  1. Visit: https://barttorvik.com/trank.php?year={year}")
            print("  2. Click 'CSV' or export button")
            print(f"  3. Save to: data/raw/barttorvik_{year}.csv")
            return None
            
    except Exception as e:
        print(f"  Error fetching Barttorvik: {e}")
        print("  Please download manually from barttorvik.com")
        return None


def fetch_kenpom(year: int = 2026, email: str = None, password: str = None) -> pd.DataFrame:
    """
    Fetch team stats from KenPom using kenpompy.
    
    Requires:
    - KenPom subscription ($24.95/year at kenpom.com)
    - pip install kenpompy
    """
    if not email or not password:
        print("KenPom requires subscription credentials.")
        print("  1. Subscribe at https://kenpom.com ($24.95/year)")
        print("  2. Run with: --email YOUR_EMAIL --password YOUR_PASSWORD")
        return None
    
    try:
        from kenpompy.utils import login
        import kenpompy.summary as kp
        import kenpompy.misc as kpm
        
        print(f"Logging into KenPom...")
        browser = login(email, password)
        
        print(f"Fetching KenPom data for {year} season...")
        
        # Get main efficiency ratings
        ratings = kpm.get_pomeroy_ratings(browser, season=str(year))
        print(f"  Got {len(ratings)} teams from ratings")
        
        # Get four factors
        four_factors = kp.get_fourfactors(browser, season=str(year))
        print(f"  Got four factors data")
        
        # Get height/experience
        height = kp.get_height(browser, season=str(year))
        print(f"  Got height/experience data")
        
        # Merge all data
        df = ratings.merge(four_factors, on='Team', how='left', suffixes=('', '_ff'))
        df = df.merge(height, on='Team', how='left', suffixes=('', '_ht'))
        
        print(f"  Successfully fetched {len(df)} teams from KenPom")
        return df
        
    except ImportError:
        print("kenpompy not installed. Run: pip install kenpompy")
        return None
    except Exception as e:
        print(f"  Error fetching KenPom: {e}")
        return None


def combine_sources(kenpom_df: pd.DataFrame, barttorvik_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine KenPom and Barttorvik data into unified format.
    
    Maps column names to match our existing data structure.
    """
    # This would need customization based on actual column names
    # For now, return the most complete source
    if kenpom_df is not None and barttorvik_df is not None:
        # Merge on team name (requires name standardization)
        print("Combining data sources...")
        # TODO: Implement merge logic with team name standardization
        return kenpom_df
    elif kenpom_df is not None:
        return kenpom_df
    else:
        return barttorvik_df


def save_data(df: pd.DataFrame, year: int, source: str):
    """Save fetched data to CSV."""
    if df is None:
        print("No data to save.")
        return
        
    output_path = PROJECT_ROOT / f"data/raw/{source}_{year}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


def print_manual_instructions(year: int = 2026):
    """Print instructions for manual data download."""
    print("\n" + "="*70)
    print("MANUAL DATA DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    print(f"""
For the {year} season, you'll need data from these sources:

1. BARTTORVIK (FREE) - Primary source for our metrics
   ------------------------------------------------
   URL: https://barttorvik.com/trank.php?year={year}
   
   Steps:
   a) Visit the URL above
   b) Scroll to bottom, look for download/export option
   c) Or use the customizable tables: 
      https://barttorvik.com/team-tables_each.php
   d) Select all metrics we need (see list below)
   e) Save as: data/raw/barttorvik_{year}.csv

2. KENPOM ($24.95/year subscription)
   ----------------------------------
   URL: https://kenpom.com/
   
   Steps:
   a) Subscribe if you haven't already
   b) Use kenpompy Python package (pip install kenpompy)
   c) Or manually copy data from the website

REQUIRED METRICS (check these columns exist):
---------------------------------------------
Core Efficiency:
  - KADJ EM (or AdjEM) - Adjusted Efficiency Margin
  - KADJ O (or AdjO)   - Adjusted Offensive Efficiency  
  - KADJ D (or AdjD)   - Adjusted Defensive Efficiency
  - BARTHAG            - Win probability metric

Tempo & Style:
  - KADJ T (or AdjT)   - Adjusted Tempo
  - EFG%, EFG%D        - Effective FG% (off/def)
  - TOV%, TOV%D        - Turnover rate (off/def)
  - OREB%, DREB%       - Rebound rates

Team Attributes:
  - TALENT             - Recruiting composite (KEY FEATURE!)
  - EXP                - Experience
  - AVG HGT            - Average height
  - ELITE SOS          - Strength of schedule

Tournament Info:
  - SEED               - Tournament seed (added after Selection Sunday)
  - CONF               - Conference

After downloading, run the merge script:
  python -m src.data.merge_season_data --year {year}
""")


def main():
    parser = argparse.ArgumentParser(description='Fetch current season college basketball data')
    parser.add_argument('--source', choices=['kenpom', 'barttorvik', 'both', 'instructions'],
                        default='instructions', help='Data source to fetch from')
    parser.add_argument('--year', type=int, default=2026, help='Season year (e.g., 2026 for 2025-26)')
    parser.add_argument('--email', type=str, help='KenPom email (for kenpom source)')
    parser.add_argument('--password', type=str, help='KenPom password (for kenpom source)')
    
    args = parser.parse_args()
    
    if args.source == 'instructions':
        print_manual_instructions(args.year)
        return
    
    print(f"\n{'='*70}")
    print(f"FETCHING {args.year} SEASON DATA")
    print(f"{'='*70}\n")
    
    kenpom_df = None
    barttorvik_df = None
    
    if args.source in ['kenpom', 'both']:
        kenpom_df = fetch_kenpom(args.year, args.email, args.password)
        if kenpom_df is not None:
            save_data(kenpom_df, args.year, 'kenpom')
    
    if args.source in ['barttorvik', 'both']:
        barttorvik_df = fetch_barttorvik(args.year)
        if barttorvik_df is not None:
            save_data(barttorvik_df, args.year, 'barttorvik')
    
    if args.source == 'both' and (kenpom_df is not None or barttorvik_df is not None):
        combined = combine_sources(kenpom_df, barttorvik_df)
        save_data(combined, args.year, 'combined')
    
    if kenpom_df is None and barttorvik_df is None:
        print("\nNo data fetched automatically. Showing manual instructions...")
        print_manual_instructions(args.year)


if __name__ == '__main__':
    main()
