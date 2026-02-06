"""
Add tournament seeds to season data after Selection Sunday.

This script can:
1. Manually add seeds from a list
2. Parse seeds from a bracket URL (NCAA.com)
3. Interactive mode to enter seeds one-by-one

Usage:
    # Interactive mode
    python add_tournament_seeds.py --year 2026

    # From a seed list file
    python add_tournament_seeds.py --year 2026 --file seeds_2026.txt

    # Quick add specific teams
    python add_tournament_seeds.py --year 2026 --add "Duke=1,UConn=1,Houston=2,..."
"""

import argparse
import pandas as pd
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent


# Historical seed data for reference (from Andy Katz's Feb 2026 projection)
# Team names matched to Barttorvik format
PROJECTED_SEEDS_2026 = {
    # 1-seeds
    'Arizona': 1, 'Michigan': 1, 'Duke': 1, 'Connecticut': 1,
    # 2-seeds  
    'Nebraska': 2, 'Houston': 2, 'Iowa St.': 2, 'Illinois': 2,
    # 3-seeds
    'Michigan St.': 3, 'Gonzaga': 3, 'Purdue': 3, 'Kansas': 3,
    # 4-seeds
    'Texas Tech': 4, 'Florida': 4, 'Vanderbilt': 4, 'BYU': 4,
    # 5-seeds
    'Virginia': 5, 'North Carolina': 5, 'Louisville': 5, 'Tennessee': 5,
    # 6-seeds
    "St. John's": 6, 'Alabama': 6, 'Arkansas': 6, 'Kentucky': 6,
    # 7-seeds
    'Iowa': 7, 'Saint Louis': 7, 'Clemson': 7, 'Auburn': 7,
    # 8-seeds
    'UCF': 8, 'Texas A&M': 8, 'Villanova': 8, 'SMU': 8,
    # 9-seeds
    'Wisconsin': 9, 'North Carolina St.': 9, 'Utah St.': 9, 'Indiana': 9,
    # 10-seeds
    "Saint Mary's": 10, 'Georgia': 10, 'USC': 10, 'Miami FL': 10,
    # 11-seeds (includes First Four)
    'UCLA': 11, 'New Mexico': 11, 'Ohio St.': 11, 'San Diego St.': 11,
    'Texas': 11, 'Miami OH': 11,
    # 12-seeds
    'Belmont': 12, 'Tulsa': 12, 'Liberty': 12, 'Yale': 12,
    # 13-seeds
    'Stephen F. Austin': 13, 'UNC Wilmington': 13, 'High Point': 13, 'Utah Valley': 13,
    # 14-seeds
    'North Dakota St.': 14, 'UC Irvine': 14, 'Troy': 14, 'Austin Peay': 14,
    # 15-seeds
    'Portland St.': 15, 'Wright St.': 15, 'Tennessee Martin': 15, 'East Tennessee St.': 15,
    # 16-seeds (includes First Four)
    'Navy': 16, 'Merrimack': 16, 'LIU Brooklyn': 16, 'Bethune Cookman': 16,
    'Vermont': 16, 'Maryland Eastern Shore': 16,
}

# Team name mappings (bracket names -> data file names)
TEAM_NAME_MAP = {
    'UConn': 'Connecticut',
    'Connecticut': 'Connecticut',
    'NC State': 'North Carolina State',
    'North Carolina St.': 'North Carolina State',
    "Saint Mary's (CA)": "Saint Mary's",
    'Miami (FL)': 'Miami',
    'Miami FL': 'Miami',
    'Miami (OH)': 'Miami OH',
    'ETSU': 'East Tennessee State',
    'SFA': 'Stephen F. Austin',
    'UNCW': 'UNC Wilmington',
    'NDSU': 'North Dakota State',
}


def load_season_data(year: int) -> tuple:
    """Load season data file."""
    # Try different file patterns
    patterns = [
        f"data/raw/KenPom Barttorvik {year}.csv",
        f"data/raw/barttorvik_{year}.csv",
        "data/raw/KenPom Barttorvik.csv",
    ]
    
    for pattern in patterns:
        path = PROJECT_ROOT / pattern
        if path.exists():
            df = pd.read_csv(path)
            if 'YEAR' in df.columns:
                df = df[df['YEAR'] == year]
            print(f"Loaded {len(df)} teams from {path}")
            return df, path
    
    print("No data file found. Please download data first:")
    print("  python -m src.data.fetch_current_season --source instructions")
    return None, None


def normalize_team_name(name: str, available_teams: set) -> str:
    """Normalize team name to match data file."""
    # Direct match
    if name in available_teams:
        return name
    
    # Check mapping
    if name in TEAM_NAME_MAP:
        mapped = TEAM_NAME_MAP[name]
        if mapped in available_teams:
            return mapped
    
    # Fuzzy match - try partial
    name_lower = name.lower()
    for team in available_teams:
        if name_lower in team.lower() or team.lower() in name_lower:
            return team
    
    return None


def add_seeds_interactive(df: pd.DataFrame) -> pd.DataFrame:
    """Interactive mode to add seeds."""
    print("\n" + "="*70)
    print("INTERACTIVE SEED ENTRY")
    print("="*70)
    print("Enter seeds in format: TeamName=Seed (e.g., Duke=1)")
    print("Type 'done' when finished, 'list' to see entered seeds")
    print("Type 'projected' to use Andy Katz's Feb 2026 projection")
    print("="*70 + "\n")
    
    available_teams = set(df['TEAM'].values)
    seeds_added = {}
    
    while True:
        entry = input("Enter seed (or 'done'/'list'/'projected'): ").strip()
        
        if entry.lower() == 'done':
            break
        elif entry.lower() == 'list':
            if seeds_added:
                print("\nSeeds entered so far:")
                for team, seed in sorted(seeds_added.items(), key=lambda x: x[1]):
                    print(f"  ({seed}) {team}")
            else:
                print("No seeds entered yet.")
            continue
        elif entry.lower() == 'projected':
            # Use projected seeds
            print("\nUsing Andy Katz's Feb 2026 bracket projection...")
            for team, seed in PROJECTED_SEEDS_2026.items():
                matched = normalize_team_name(team, available_teams)
                if matched:
                    seeds_added[matched] = seed
            print(f"Added {len(seeds_added)} projected seeds.")
            break
        
        # Parse entry
        match = re.match(r'(.+?)\s*=\s*(\d+)', entry)
        if not match:
            print("Invalid format. Use: TeamName=Seed")
            continue
        
        team_name, seed = match.groups()
        seed = int(seed)
        
        if seed < 1 or seed > 16:
            print("Seed must be 1-16")
            continue
        
        # Find team in data
        matched = normalize_team_name(team_name, available_teams)
        if matched:
            seeds_added[matched] = seed
            print(f"  Added: ({seed}) {matched}")
        else:
            print(f"  Team not found: {team_name}")
            print(f"  Similar teams: {[t for t in available_teams if team_name.lower()[:4] in t.lower()][:5]}")
    
    # Apply seeds to dataframe
    if 'SEED' not in df.columns:
        df['SEED'] = None
    
    for team, seed in seeds_added.items():
        df.loc[df['TEAM'] == team, 'SEED'] = seed
    
    print(f"\nApplied {len(seeds_added)} seeds to data.")
    return df


def add_seeds_from_dict(df: pd.DataFrame, seeds: dict) -> pd.DataFrame:
    """Add seeds from a dictionary."""
    available_teams = set(df['TEAM'].values)
    
    if 'SEED' not in df.columns:
        df['SEED'] = None
    
    matched = 0
    for team, seed in seeds.items():
        team_matched = normalize_team_name(team, available_teams)
        if team_matched:
            df.loc[df['TEAM'] == team_matched, 'SEED'] = seed
            matched += 1
        else:
            print(f"  Warning: Team not found - {team}")
    
    print(f"Applied {matched}/{len(seeds)} seeds")
    return df


def add_seeds_from_string(df: pd.DataFrame, seed_string: str) -> pd.DataFrame:
    """Parse seeds from comma-separated string like 'Duke=1,UConn=1,...'"""
    seeds = {}
    for item in seed_string.split(','):
        match = re.match(r'(.+?)\s*=\s*(\d+)', item.strip())
        if match:
            team, seed = match.groups()
            seeds[team] = int(seed)
    
    return add_seeds_from_dict(df, seeds)


def show_tournament_field(df: pd.DataFrame):
    """Display the tournament field by seed."""
    tourney = df[df['SEED'].notna()].copy()
    tourney['SEED'] = tourney['SEED'].astype(int)
    
    print("\n" + "="*70)
    print("2026 NCAA TOURNAMENT FIELD")
    print("="*70)
    
    for seed in range(1, 17):
        teams = tourney[tourney['SEED'] == seed]['TEAM'].tolist()
        if teams:
            print(f"\n{seed}-seeds: {', '.join(teams)}")
    
    print(f"\nTotal tournament teams: {len(tourney)}")


def main():
    parser = argparse.ArgumentParser(description='Add tournament seeds to season data')
    parser.add_argument('--year', type=int, default=2026, help='Season year')
    parser.add_argument('--add', type=str, help='Comma-separated seeds: "Duke=1,UConn=1,..."')
    parser.add_argument('--projected', action='store_true', help='Use projected seeds (Feb 2026)')
    parser.add_argument('--show', action='store_true', help='Show current tournament field')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"NCAA TOURNAMENT SEED MANAGER - {args.year}")
    print(f"{'='*70}")
    
    # Load data
    df, source_path = load_season_data(args.year)
    if df is None:
        return
    
    if args.show:
        show_tournament_field(df)
        return
    
    # Add seeds
    if args.add:
        df = add_seeds_from_string(df, args.add)
    elif args.projected:
        df = add_seeds_from_dict(df, PROJECTED_SEEDS_2026)
    else:
        df = add_seeds_interactive(df)
    
    # Show result
    show_tournament_field(df)
    
    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / f"data/raw/KenPom Barttorvik {args.year}.csv"
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    print("\nNext steps:")
    print(f"  python predict_games.py --year {args.year}")
    print(f"  python main.py --year {args.year}")


if __name__ == '__main__':
    main()
