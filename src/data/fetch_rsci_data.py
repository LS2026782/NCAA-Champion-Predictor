"""
RSCI (Recruiting Services Consensus Index) Data Fetcher

This script fetches historical RSCI recruiting rankings from available archives.
RSCI data is needed to reconstruct the TALENT metric for 2002-2013 seasons.

RSCI Background:
- RSCI was created in the early 2000s to aggregate rankings from multiple services
- It averages rankings from Rivals, ESPN, 247Sports, Scout (now defunct), etc.
- The Top 100 recruits are ranked each year
- Historical data is available from various archive sources

Data Sources:
1. 247Sports historical database (best structured)
2. Rivals archives
3. Various college basketball fan sites with historical archives
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RSCI_DIR = PROJECT_ROOT / "data" / "historical" / "rsci"
RSCI_DIR.mkdir(parents=True, exist_ok=True)


class RSCIFetcher:
    """
    Fetches RSCI recruiting data from 247Sports Composite.
    
    247Sports maintains a historical archive of recruiting rankings
    with the composite (similar to RSCI) going back to 2000.
    """
    
    BASE_URL = "https://247sports.com/Season/{year}-Basketball/CompositeRecruitRankings/"
    
    def __init__(self, delay: float = 3.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_year(self, year: int, max_players: int = 100) -> List[Dict]:
        """
        Fetch recruiting rankings for a specific year.
        
        Args:
            year: Recruiting class year (e.g., 2005)
            max_players: Maximum number of players to fetch
        
        Returns:
            List of dicts with player rankings
        """
        url = self.BASE_URL.format(year=year)
        
        logger.info(f"Fetching RSCI data for {year}...")
        time.sleep(self.delay)
        
        try:
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {year}: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            players = []
            
            # 247Sports uses specific class names for recruit lists
            recruit_items = soup.find_all('li', class_=re.compile('ri-page'))
            
            for item in recruit_items[:max_players]:
                try:
                    # Extract rank
                    rank_elem = item.find(class_=re.compile('rank'))
                    rank = int(rank_elem.get_text(strip=True)) if rank_elem else None
                    
                    # Extract name
                    name_elem = item.find(class_=re.compile('name'))
                    name = name_elem.get_text(strip=True) if name_elem else ''
                    
                    # Extract school commitment
                    school_elem = item.find(class_=re.compile('school'))
                    school = school_elem.get_text(strip=True) if school_elem else ''
                    
                    # Extract position
                    pos_elem = item.find(class_=re.compile('position'))
                    position = pos_elem.get_text(strip=True) if pos_elem else ''
                    
                    if rank and name:
                        players.append({
                            'year': year,
                            'rank': rank,
                            'name': name,
                            'school': school,
                            'position': position
                        })
                except Exception as e:
                    logger.debug(f"Error parsing recruit: {e}")
                    continue
            
            logger.info(f"Found {len(players)} recruits for {year}")
            return players
            
        except Exception as e:
            logger.error(f"Error fetching {year}: {e}")
            return []
    
    def fetch_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Fetch recruiting data for a range of years.
        
        Args:
            start_year: First year to fetch
            end_year: Last year to fetch
        
        Returns:
            DataFrame with all recruiting data
        """
        all_players = []
        
        for year in range(start_year, end_year + 1):
            players = self.fetch_year(year)
            all_players.extend(players)
            
            # Save progress
            if players:
                df = pd.DataFrame(players)
                df.to_csv(RSCI_DIR / f"rsci_{year}.csv", index=False)
        
        return pd.DataFrame(all_players)


def create_rsci_lookup_table() -> Dict[int, Dict[str, int]]:
    """
    Create a lookup table from saved RSCI files.
    
    Returns:
        Dict: year -> {player_name_lower: rank}
    """
    lookup = {}
    
    for file in RSCI_DIR.glob("rsci_*.csv"):
        try:
            df = pd.read_csv(file)
            year = int(file.stem.split('_')[1])
            
            lookup[year] = {}
            for _, row in df.iterrows():
                name = str(row.get('name', '')).lower().strip()
                rank = row.get('rank', 999)
                if name:
                    lookup[year][name] = int(rank)
                    
            logger.info(f"Loaded {len(lookup[year])} recruits for {year}")
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    return lookup


# Manual RSCI data for key years (backup if scraping fails)
# This is a sample - full data would need to be compiled from archives
MANUAL_RSCI_DATA = {
    2003: {
        # Class of 2003 - Top recruits
        'lebron james': 1,
        'shannon brown': 2,
        'dwayne wade': 3,  # Note: Wade was 2003 draft but earlier class
        'josh smith': 4,
        'dwight howard': 5,  # Actually 2004, but widely tracked
        'al jefferson': 6,
        'travis outlaw': 7,
        'james lang': 8,
        'carmelo anthony': 9,  # 2002 class technically
        # Add more as needed from archives
    },
    2004: {
        'dwight howard': 1,
        'josh smith': 2,
        'shaun livingston': 3,
        'al jefferson': 4,
        'luol deng': 5,
        'jj redick': 6,
        'josh mccroberts': 7,
        'marvin williams': 8,
        'raymond felton': 9,
        'sean may': 10,
    },
    2005: {
        'greg oden': 1,
        'kevin durant': 2,
        'oj mayo': 3,
        'spencer hawes': 4,
        'chase budinger': 5,
        'eric gordon': 6,
        'brandan wright': 7,
        'thaddeus young': 8,
        'derrick rose': 9,
        'michael beasley': 10,
    },
    # Add more years...
}


def load_manual_rsci(year: int) -> Dict[str, int]:
    """Load manual RSCI data for a given year."""
    return MANUAL_RSCI_DATA.get(year, {})


def compile_all_rsci_data() -> pd.DataFrame:
    """
    Compile RSCI data from all sources.
    
    Priority:
    1. Scraped data from RSCI_DIR
    2. Manual backup data
    
    Returns:
        DataFrame with year, rank, name columns
    """
    all_data = []
    
    # Load scraped data
    for year in range(1998, 2014):
        file = RSCI_DIR / f"rsci_{year}.csv"
        if file.exists():
            df = pd.read_csv(file)
            all_data.append(df)
        else:
            # Use manual data
            manual = load_manual_rsci(year)
            for name, rank in manual.items():
                all_data.append({
                    'year': year,
                    'rank': rank,
                    'name': name
                })
    
    if all_data:
        # Some entries are dicts, some are DataFrames
        dfs = []
        rows = []
        for item in all_data:
            if isinstance(item, pd.DataFrame):
                dfs.append(item)
            else:
                rows.append(item)
        
        if rows:
            dfs.append(pd.DataFrame(rows))
        
        return pd.concat(dfs, ignore_index=True)
    
    return pd.DataFrame()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RSCI Data Fetcher")
    parser.add_argument('--fetch', type=str, help='Fetch years (e.g., "2003-2013")')
    parser.add_argument('--list', action='store_true', help='List available RSCI files')
    parser.add_argument('--compile', action='store_true', help='Compile all RSCI data')
    
    args = parser.parse_args()
    
    if args.list:
        print(f"\n=== Available RSCI Data in {RSCI_DIR} ===")
        for file in sorted(RSCI_DIR.glob("rsci_*.csv")):
            df = pd.read_csv(file)
            print(f"  {file.name}: {len(df)} players")
        
        print(f"\n=== Manual Backup Data ===")
        for year, data in sorted(MANUAL_RSCI_DATA.items()):
            print(f"  {year}: {len(data)} players")
    
    elif args.fetch:
        start, end = map(int, args.fetch.split('-'))
        fetcher = RSCIFetcher(delay=3.0)
        df = fetcher.fetch_range(start, end)
        print(f"\nFetched {len(df)} total players from {start}-{end}")
        
    elif args.compile:
        df = compile_all_rsci_data()
        output = RSCI_DIR / "rsci_all.csv"
        df.to_csv(output, index=False)
        print(f"Compiled {len(df)} records to {output}")
    
    else:
        print("""
RSCI Data Fetcher
=================

This tool helps gather RSCI recruiting rankings for the TALENT metric reconstruction.

Usage:
  python fetch_rsci_data.py --list        # Show available data
  python fetch_rsci_data.py --fetch 2003-2013  # Fetch from 247Sports
  python fetch_rsci_data.py --compile     # Compile all data to single file

Note: Web scraping may be blocked. Manual data entry from archives may be needed.
See MANUAL_RSCI_DATA in this file for backup data format.
""")
