"""
Historical Data Reconstruction Pipeline (2002-2007)

This module implements the reconstruction strategy outlined in the 
"LONGITUDINAL ARCHITECTURE FOR PREDICTIVE MODELING" document.

The goal is to extend training data back to 2002-2003 by:
1. Using KenPom efficiency data as the base (AdjO, AdjD, AdjEM, Four Factors)
2. Calculating BARTHAG from the efficiency formula
3. Calculating ELITE SOS from opponent quality
4. Reconstructing TALENT from RSCI recruiting data weighted by minutes
5. Reconstructing EXP from roster class data weighted by minutes
6. Reconstructing HEIGHT metrics from roster data

Data Sources:
- KenPom ($25 subscription) - Efficiency metrics back to 2002
- Sports-Reference - Free roster data (class, height, minutes)
- RSCI Archives - Historical recruiting rankings (1998-2013)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
import re
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
HISTORICAL_DIR = DATA_DIR / "historical"

# Ensure directories exist
HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# BARTHAG CALCULATION
# =============================================================================

def calculate_barthag(adj_o: float, adj_d: float, exponent: float = 11.5) -> float:
    """
    Calculate BARTHAG (win probability vs average D1 team on neutral court).
    
    Formula: BARTHAG = AdjO^11.5 / (AdjO^11.5 + AdjD^11.5)
    
    This is based on the Pythagorean expectation adapted for college basketball.
    The exponent of 11.5 has been found optimal for NCAA variance.
    
    Args:
        adj_o: Adjusted Offensive Efficiency (points per 100 possessions)
        adj_d: Adjusted Defensive Efficiency (points per 100 possessions)
        exponent: Pythagorean exponent (default 11.5 for college basketball)
    
    Returns:
        Win probability (0.0 to 1.0)
    """
    if pd.isna(adj_o) or pd.isna(adj_d):
        return np.nan
    
    adj_o_exp = adj_o ** exponent
    adj_d_exp = adj_d ** exponent
    
    return adj_o_exp / (adj_o_exp + adj_d_exp)


def add_barthag_to_dataframe(df: pd.DataFrame, 
                             adj_o_col: str = 'KADJ O',
                             adj_d_col: str = 'KADJ D') -> pd.DataFrame:
    """
    Add calculated BARTHAG column to a dataframe.
    
    Args:
        df: DataFrame with efficiency columns
        adj_o_col: Name of adjusted offensive efficiency column
        adj_d_col: Name of adjusted defensive efficiency column
    
    Returns:
        DataFrame with BARTHAG column added
    """
    df = df.copy()
    df['BARTHAG'] = df.apply(
        lambda row: calculate_barthag(row[adj_o_col], row[adj_d_col]),
        axis=1
    )
    return df


# =============================================================================
# TALENT RECONSTRUCTION (RSCI-based)
# =============================================================================

def calculate_talent_score(rank: int, sigma: float = 25.0) -> float:
    """
    Calculate talent score for a single player using exponential decay.
    
    Formula: 100 * e^(-(rank-1)/sigma)
    
    This creates a power-law distribution where:
    - Rank 1 = 100 points
    - Rank 10 ≈ 70 points
    - Rank 50 ≈ 14 points
    - Rank 100 ≈ 2 points
    - Unranked = 0.5 points (replacement level)
    
    Args:
        rank: RSCI ranking (1-100, or 0/None for unranked)
        sigma: Decay constant (default 25)
    
    Returns:
        Talent score (0.5 to 100)
    """
    if rank is None or rank <= 0 or rank > 100:
        return 0.5  # Replacement level for unranked players
    
    return 100 * np.exp(-(rank - 1) / sigma)


def calculate_team_talent(roster: List[Dict], minutes_col: str = 'minutes') -> float:
    """
    Calculate team talent score as minutes-weighted average.
    
    Formula: SUM(talent_score * minutes) / SUM(minutes)
    
    Args:
        roster: List of dicts with 'rsci_rank' and 'minutes' keys
        minutes_col: Column name for minutes
    
    Returns:
        Team talent score (weighted average)
    """
    if not roster:
        return np.nan
    
    total_weighted = 0.0
    total_minutes = 0.0
    
    for player in roster:
        minutes = player.get(minutes_col, 0)
        if minutes <= 0:
            continue
            
        talent = calculate_talent_score(player.get('rsci_rank'))
        total_weighted += talent * minutes
        total_minutes += minutes
    
    if total_minutes == 0:
        return np.nan
    
    return total_weighted / total_minutes


# =============================================================================
# EXPERIENCE RECONSTRUCTION
# =============================================================================

# Class value mapping
CLASS_VALUES = {
    'fr': 0, 'freshman': 0,
    'so': 1, 'sophomore': 1,
    'jr': 2, 'junior': 2,
    'sr': 3, 'senior': 3,
    'gr': 3, 'graduate': 3,  # Grad students count as seniors
}


def calculate_team_experience(roster: List[Dict], minutes_col: str = 'minutes') -> float:
    """
    Calculate team experience as minutes-weighted class average.
    
    Formula: SUM(class_value * minutes) / SUM(minutes)
    
    Where class_value: Fr=0, So=1, Jr=2, Sr=3
    
    Args:
        roster: List of dicts with 'class' and 'minutes' keys
        minutes_col: Column name for minutes
    
    Returns:
        Team experience score (0.0 to 3.0)
    """
    if not roster:
        return np.nan
    
    total_weighted = 0.0
    total_minutes = 0.0
    
    for player in roster:
        minutes = player.get(minutes_col, 0)
        if minutes <= 0:
            continue
        
        player_class = str(player.get('class', '')).lower().strip()
        class_value = CLASS_VALUES.get(player_class, 1.5)  # Default to mid-range
        
        total_weighted += class_value * minutes
        total_minutes += minutes
    
    if total_minutes == 0:
        return np.nan
    
    return total_weighted / total_minutes


# =============================================================================
# HEIGHT RECONSTRUCTION
# =============================================================================

def parse_height_to_inches(height_str: str) -> Optional[float]:
    """
    Convert height string to inches.
    
    Handles formats: "6-8", "6'8\"", "6-8", "80" (already inches)
    
    Args:
        height_str: Height string in various formats
    
    Returns:
        Height in inches, or None if unparseable
    """
    if pd.isna(height_str):
        return None
    
    height_str = str(height_str).strip()
    
    # Already numeric (inches)
    if height_str.replace('.', '').isdigit():
        return float(height_str)
    
    # Pattern: 6-8, 6'8", 6'8, 6-8
    match = re.match(r"(\d+)['\-](\d+)", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 12 + inches
    
    return None


def calculate_effective_height(roster: List[Dict], 
                                 top_percent: float = 0.40) -> float:
    """
    Calculate effective height (avg height of interior players).
    
    This measures the size of players getting the most "big man" minutes,
    defined as the tallest players comprising top 40% of minutes.
    
    Args:
        roster: List of dicts with 'height' (inches) and 'minutes' keys
        top_percent: Fraction of minutes for "interior" definition (default 0.40)
    
    Returns:
        Average height (inches) of top interior minute-getters
    """
    if not roster:
        return np.nan
    
    # Filter to players with valid height and minutes
    valid_players = []
    for p in roster:
        height = parse_height_to_inches(p.get('height'))
        minutes = p.get('minutes', 0)
        if height and minutes > 0:
            valid_players.append({'height': height, 'minutes': minutes})
    
    if not valid_players:
        return np.nan
    
    # Sort by height descending
    valid_players.sort(key=lambda x: x['height'], reverse=True)
    
    # Accumulate minutes until we hit the threshold
    total_minutes = sum(p['minutes'] for p in valid_players)
    target_minutes = total_minutes * top_percent
    
    accumulated = 0
    interior_players = []
    
    for player in valid_players:
        interior_players.append(player)
        accumulated += player['minutes']
        if accumulated >= target_minutes:
            break
    
    if not interior_players:
        return np.nan
    
    # Calculate weighted average height of interior players
    total_height_minutes = sum(p['height'] * p['minutes'] for p in interior_players)
    total_int_minutes = sum(p['minutes'] for p in interior_players)
    
    return total_height_minutes / total_int_minutes


def calculate_average_height(roster: List[Dict]) -> float:
    """
    Calculate average height weighted by minutes.
    
    Args:
        roster: List of dicts with 'height' (inches) and 'minutes' keys
    
    Returns:
        Minutes-weighted average height (inches)
    """
    if not roster:
        return np.nan
    
    total_weighted = 0.0
    total_minutes = 0.0
    
    for player in roster:
        height = parse_height_to_inches(player.get('height'))
        minutes = player.get('minutes', 0)
        
        if height and minutes > 0:
            total_weighted += height * minutes
            total_minutes += minutes
    
    if total_minutes == 0:
        return np.nan
    
    return total_weighted / total_minutes


# =============================================================================
# ELITE SOS CALCULATION
# =============================================================================

def calculate_elite_sos(games: List[Dict], 
                        elite_threshold_rank: int = 50) -> float:
    """
    Calculate Elite Strength of Schedule.
    
    This is the average AdjEM of opponents ranked in the Top 50.
    
    Args:
        games: List of game dicts with 'opponent_adj_em' and 'opponent_rank'
        elite_threshold_rank: Rank cutoff for "elite" opponents (default 50)
    
    Returns:
        Average AdjEM of elite opponents
    """
    if not games:
        return np.nan
    
    elite_games = [
        g for g in games 
        if g.get('opponent_rank', 999) <= elite_threshold_rank
        and not pd.isna(g.get('opponent_adj_em'))
    ]
    
    if not elite_games:
        return 0.0  # No elite opponents
    
    return np.mean([g['opponent_adj_em'] for g in elite_games])


# =============================================================================
# SPORTS-REFERENCE SCRAPER
# =============================================================================

class SportsReferenceRosterScraper:
    """
    Scraper for Sports-Reference college basketball roster data.
    
    Provides: Player name, class (Fr/So/Jr/Sr), height, position, minutes
    """
    
    BASE_URL = "https://www.sports-reference.com/cbb/schools"
    
    # Team name mapping to Sports-Reference slugs
    TEAM_SLUGS = {
        # Will need to populate this mapping
        # Format: 'KenPom Name': 'sports-reference-slug'
        'Duke': 'duke',
        'North Carolina': 'north-carolina',
        'Kentucky': 'kentucky',
        'Kansas': 'kansas',
        'Gonzaga': 'gonzaga',
        'UConn': 'connecticut',
        'Connecticut': 'connecticut',
        'Michigan St.': 'michigan-state',
        'Michigan State': 'michigan-state',
        'Syracuse': 'syracuse',
        'Louisville': 'louisville',
        'UCLA': 'ucla',
        'Arizona': 'arizona',
        'Florida': 'florida',
        'Ohio St.': 'ohio-state',
        'Ohio State': 'ohio-state',
        'Villanova': 'villanova',
        'Baylor': 'baylor',
        'Indiana': 'indiana',
        'Wisconsin': 'wisconsin',
        'Maryland': 'maryland',
        # Add more as needed...
    }
    
    def __init__(self, delay: float = 3.0):
        """
        Initialize scraper with rate limiting.
        
        Args:
            delay: Seconds to wait between requests (be respectful!)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_team_slug(self, team_name: str) -> Optional[str]:
        """Get Sports-Reference slug for a team name."""
        if team_name in self.TEAM_SLUGS:
            return self.TEAM_SLUGS[team_name]
        
        # Try to generate slug
        slug = team_name.lower()
        slug = re.sub(r'[^a-z\s]', '', slug)
        slug = slug.replace(' ', '-')
        return slug
    
    def scrape_roster(self, team_name: str, year: int) -> Optional[List[Dict]]:
        """
        Scrape roster for a team-year.
        
        Args:
            team_name: Team name (KenPom format)
            year: Season year (e.g., 2005 for 2004-05 season)
        
        Returns:
            List of player dicts, or None if failed
        """
        slug = self.get_team_slug(team_name)
        if not slug:
            logger.warning(f"No slug found for {team_name}")
            return None
        
        # Sports-Ref uses year for the spring semester
        url = f"{self.BASE_URL}/{slug}/{year}.html"
        
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the roster table
            roster_table = soup.find('table', {'id': 'roster'})
            if not roster_table:
                logger.warning(f"No roster table found for {team_name} {year}")
                return None
            
            roster = []
            rows = roster_table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) < 4:
                    continue
                
                player = {
                    'name': cols[1].get_text(strip=True) if len(cols) > 1 else '',
                    'class': cols[2].get_text(strip=True) if len(cols) > 2 else '',
                    'position': cols[3].get_text(strip=True) if len(cols) > 3 else '',
                    'height': cols[4].get_text(strip=True) if len(cols) > 4 else '',
                }
                roster.append(player)
            
            return roster
            
        except Exception as e:
            logger.error(f"Error scraping {team_name} {year}: {e}")
            return None
    
    def scrape_season_stats(self, team_name: str, year: int) -> Optional[List[Dict]]:
        """
        Scrape per-game stats (for minutes) for a team-year.
        
        Args:
            team_name: Team name (KenPom format)
            year: Season year
        
        Returns:
            List of player dicts with 'name' and 'minutes', or None
        """
        slug = self.get_team_slug(team_name)
        if not slug:
            return None
        
        url = f"{self.BASE_URL}/{slug}/{year}.html"
        
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find per-game stats table
            stats_table = soup.find('table', {'id': 'per_game'})
            if not stats_table:
                return None
            
            players = []
            rows = stats_table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) < 5:
                    continue
                
                # Find player name and minutes columns
                name_col = row.find('td', {'data-stat': 'player'})
                mp_col = row.find('td', {'data-stat': 'mp_per_g'})
                games_col = row.find('td', {'data-stat': 'g'})
                
                if name_col and mp_col and games_col:
                    try:
                        name = name_col.get_text(strip=True)
                        mp_per_game = float(mp_col.get_text(strip=True) or 0)
                        games = int(games_col.get_text(strip=True) or 0)
                        total_minutes = mp_per_game * games
                        
                        players.append({
                            'name': name,
                            'minutes': total_minutes
                        })
                    except (ValueError, AttributeError):
                        continue
            
            return players
            
        except Exception as e:
            logger.error(f"Error scraping stats for {team_name} {year}: {e}")
            return None


# =============================================================================
# RSCI DATA HANDLER
# =============================================================================

class RSCIDataHandler:
    """
    Handler for RSCI (Recruiting Services Consensus Index) data.
    
    RSCI data is available from various archive sources and provides
    the composite recruiting rankings used to calculate TALENT.
    """
    
    def __init__(self):
        self.rsci_cache: Dict[int, Dict[str, int]] = {}  # year -> {name: rank}
    
    def load_rsci_year(self, year: int, filepath: Path) -> Dict[str, int]:
        """
        Load RSCI rankings for a given recruiting class year.
        
        Args:
            year: Recruiting class year (e.g., 2003 for class of 2003)
            filepath: Path to CSV/text file with rankings
        
        Returns:
            Dict mapping player name to rank
        """
        rankings = {}
        
        try:
            # Try different formats
            df = pd.read_csv(filepath)
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Find name and rank columns
            name_col = next((c for c in df.columns if 'name' in c or 'player' in c), None)
            rank_col = next((c for c in df.columns if 'rank' in c or 'rsci' in c), None)
            
            if name_col and rank_col:
                for _, row in df.iterrows():
                    name = str(row[name_col]).strip()
                    try:
                        rank = int(row[rank_col])
                        rankings[name.lower()] = rank
                    except (ValueError, TypeError):
                        continue
            
            self.rsci_cache[year] = rankings
            logger.info(f"Loaded {len(rankings)} RSCI rankings for {year}")
            
        except Exception as e:
            logger.error(f"Error loading RSCI data for {year}: {e}")
        
        return rankings
    
    def get_player_rank(self, name: str, year: int) -> Optional[int]:
        """
        Get RSCI rank for a player.
        
        Args:
            name: Player name
            year: Recruiting class year
        
        Returns:
            RSCI rank (1-100) or None if not ranked
        """
        if year not in self.rsci_cache:
            return None
        
        rankings = self.rsci_cache[year]
        name_lower = name.lower().strip()
        
        # Exact match
        if name_lower in rankings:
            return rankings[name_lower]
        
        # Fuzzy match (simplified)
        for ranked_name, rank in rankings.items():
            # Check if last names match
            if name_lower.split()[-1] == ranked_name.split()[-1]:
                # Check first initial
                if name_lower[0] == ranked_name[0]:
                    return rank
        
        return None


# =============================================================================
# MAIN RECONSTRUCTION PIPELINE
# =============================================================================

def reconstruct_historical_season(year: int,
                                   kenpom_data: pd.DataFrame,
                                   rsci_handler: RSCIDataHandler,
                                   roster_scraper: SportsReferenceRosterScraper,
                                   teams: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Reconstruct full feature set for a historical season.
    
    Args:
        year: Season year (e.g., 2005 for 2004-05)
        kenpom_data: DataFrame with KenPom efficiency data
        rsci_handler: RSCI data handler
        roster_scraper: Sports-Reference scraper
        teams: Optional list of teams to process (None = all)
    
    Returns:
        DataFrame with reconstructed features
    """
    results = []
    
    season_data = kenpom_data[kenpom_data['YEAR'] == year].copy()
    
    if teams:
        season_data = season_data[season_data['TEAM'].isin(teams)]
    
    for _, team_row in season_data.iterrows():
        team_name = team_row['TEAM']
        
        # Start with existing KenPom data
        team_result = team_row.to_dict()
        
        # Calculate BARTHAG from efficiency (if not already present)
        if pd.isna(team_row.get('BARTHAG')):
            team_result['BARTHAG'] = calculate_barthag(
                team_row.get('KADJ O', team_row.get('ADJ O')),
                team_row.get('KADJ D', team_row.get('ADJ D'))
            )
        
        # Try to get roster data
        roster_data = roster_scraper.scrape_roster(team_name, year)
        stats_data = roster_scraper.scrape_season_stats(team_name, year)
        
        if roster_data and stats_data:
            # Merge roster with stats (by name)
            stats_map = {p['name'].lower(): p['minutes'] for p in stats_data}
            
            full_roster = []
            for player in roster_data:
                player_full = player.copy()
                player_full['minutes'] = stats_map.get(player['name'].lower(), 0)
                
                # Look up RSCI rank (estimate recruiting year from class)
                class_val = CLASS_VALUES.get(player['class'].lower(), 1)
                recruit_year = year - class_val
                player_full['rsci_rank'] = rsci_handler.get_player_rank(
                    player['name'], recruit_year
                )
                
                full_roster.append(player_full)
            
            # Calculate reconstructed metrics
            team_result['TALENT'] = calculate_team_talent(full_roster)
            team_result['EXP'] = calculate_team_experience(full_roster)
            team_result['AVG HGT'] = calculate_average_height(full_roster)
            team_result['EFF HGT'] = calculate_effective_height(full_roster)
        
        results.append(team_result)
        logger.info(f"Processed {team_name} {year}")
    
    return pd.DataFrame(results)


def create_unified_dataset(start_year: int = 2002,
                           end_year: int = 2025,
                           kenpom_file: Optional[Path] = None,
                           output_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Create a unified dataset spanning all years with consistent features.
    
    This is the main entry point for reconstruction.
    
    Args:
        start_year: First season to include
        end_year: Last season to include
        kenpom_file: Path to KenPom data file
        output_file: Path to save output
    
    Returns:
        Unified DataFrame with all years
    """
    logger.info(f"Creating unified dataset for {start_year}-{end_year}")
    
    # Load existing data
    if kenpom_file is None:
        kenpom_file = RAW_DATA_DIR / "KenPom Barttorvik.csv"
    
    existing_data = pd.read_csv(kenpom_file)
    existing_years = set(existing_data['YEAR'].unique())
    
    logger.info(f"Existing data years: {sorted(existing_years)}")
    
    # Identify years needing reconstruction
    all_years = set(range(start_year, end_year + 1)) - {2020}  # Exclude COVID year
    missing_years = all_years - existing_years
    
    logger.info(f"Years needing reconstruction: {sorted(missing_years)}")
    
    if not missing_years:
        logger.info("No reconstruction needed - all years present")
        return existing_data
    
    # Initialize scrapers
    rsci_handler = RSCIDataHandler()
    roster_scraper = SportsReferenceRosterScraper(delay=3.0)
    
    # Reconstruct missing years
    reconstructed_dfs = []
    
    for year in sorted(missing_years):
        logger.info(f"Reconstructing {year}...")
        # This would require KenPom historical data to be available
        # For now, this is a placeholder for the reconstruction logic
        logger.warning(f"Year {year} needs KenPom base data to reconstruct")
    
    # Combine all data
    all_dfs = [existing_data] + reconstructed_dfs
    unified_df = pd.concat(all_dfs, ignore_index=True)
    unified_df = unified_df.sort_values(['YEAR', 'TEAM']).reset_index(drop=True)
    
    if output_file:
        unified_df.to_csv(output_file, index=False)
        logger.info(f"Saved unified dataset to {output_file}")
    
    return unified_df


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Historical Data Reconstruction Pipeline"
    )
    parser.add_argument(
        '--test-barthag', 
        action='store_true',
        help='Test BARTHAG calculation on sample data'
    )
    parser.add_argument(
        '--test-talent',
        action='store_true', 
        help='Test TALENT calculation on sample roster'
    )
    parser.add_argument(
        '--scrape-team',
        type=str,
        help='Scrape roster for a specific team (format: "Team Name,Year")'
    )
    
    args = parser.parse_args()
    
    if args.test_barthag:
        print("\n=== BARTHAG Calculation Test ===")
        # Sample teams from typical efficiency ratings
        samples = [
            ("Elite Team", 120.0, 95.0),    # +25 AdjEM
            ("Good Team", 110.0, 100.0),    # +10 AdjEM
            ("Average Team", 105.0, 105.0), # 0 AdjEM
            ("Below Avg", 100.0, 110.0),    # -10 AdjEM
            ("Weak Team", 95.0, 115.0),     # -20 AdjEM
        ]
        
        for name, adj_o, adj_d in samples:
            barthag = calculate_barthag(adj_o, adj_d)
            adj_em = adj_o - adj_d
            print(f"{name:15s} | AdjO: {adj_o:5.1f} | AdjD: {adj_d:5.1f} | "
                  f"AdjEM: {adj_em:+5.1f} | BARTHAG: {barthag:.4f}")
    
    if args.test_talent:
        print("\n=== TALENT Calculation Test ===")
        # Sample roster with RSCI ranks
        sample_roster = [
            {'name': 'Star Player', 'rsci_rank': 3, 'minutes': 1000},
            {'name': 'Starter 2', 'rsci_rank': 25, 'minutes': 800},
            {'name': 'Starter 3', 'rsci_rank': None, 'minutes': 750},  # Unranked
            {'name': 'Starter 4', 'rsci_rank': 75, 'minutes': 700},
            {'name': 'Starter 5', 'rsci_rank': None, 'minutes': 650},  # Unranked
            {'name': 'Bench 1', 'rsci_rank': None, 'minutes': 300},    # Unranked
            {'name': 'Bench 2', 'rsci_rank': None, 'minutes': 200},    # Unranked
        ]
        
        print("Sample Roster:")
        for p in sample_roster:
            talent = calculate_talent_score(p.get('rsci_rank'))
            print(f"  {p['name']:15s} | Rank: {str(p.get('rsci_rank') or 'Unranked'):8s} | "
                  f"Minutes: {p['minutes']:4d} | Score: {talent:.2f}")
        
        team_talent = calculate_team_talent(sample_roster)
        print(f"\nTeam TALENT Score: {team_talent:.2f}")
    
    if args.scrape_team:
        team_name, year = args.scrape_team.split(',')
        year = int(year.strip())
        team_name = team_name.strip()
        
        print(f"\n=== Scraping {team_name} {year} ===")
        scraper = SportsReferenceRosterScraper(delay=1.0)
        roster = scraper.scrape_roster(team_name, year)
        
        if roster:
            print(f"Found {len(roster)} players:")
            for p in roster:
                print(f"  {p}")
        else:
            print("Failed to scrape roster")
