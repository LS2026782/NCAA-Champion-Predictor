#!/usr/bin/env python
"""
Add historical tournament results (ROUND column) to the unified dataset.

The ROUND column encodes tournament progression:
  1  = Champion (won the tournament)
  2  = Runner-up (lost in championship game)
  4  = Final Four (lost in semifinal)
  8  = Elite Eight
  16 = Sweet Sixteen
  32 = Round of 32
  64 = Round of 64 (first round loss)
  68 = First Four (play-in game loss)

Lower number = further in tournament = better performance.

Sources: NCAA official records, Sports Reference
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"

# Historical NCAA Tournament Champions and Runners-Up (2002-2024)
# Format: year -> {team: round}
TOURNAMENT_RESULTS = {
    2002: {
        'Maryland': 1, 'Indiana': 2,
        'Kansas': 4, 'Oklahoma': 4,
        'Oregon': 8, 'Connecticut': 8, 'Kent St.': 8, 'Missouri': 8,
    },
    2003: {
        'Syracuse': 1, 'Kansas': 2,
        'Texas': 4, 'Marquette': 4,
        'Oklahoma': 8, 'Auburn': 8, 'Notre Dame': 8, 'Kentucky': 8,
    },
    2004: {
        'Connecticut': 1, 'Georgia Tech': 2,
        'Duke': 4, 'Oklahoma St.': 4,
        'Alabama': 8, 'Xavier': 8, 'Kansas': 8, 'Texas': 8,
    },
    2005: {
        'North Carolina': 1, 'Illinois': 2,
        'Michigan St.': 4, 'Louisville': 4,
        'Arizona': 8, 'Kentucky': 8, 'West Virginia': 8, 'Wisconsin': 8,
    },
    2006: {
        'Florida': 1, 'UCLA': 2,
        'George Mason': 4, 'LSU': 4,
        'Villanova': 8, 'Connecticut': 8, 'Texas': 8, 'Gonzaga': 8,
    },
    2007: {
        'Florida': 1, 'Ohio St.': 2,
        'UCLA': 4, 'Georgetown': 4,
        'Kansas': 8, 'North Carolina': 8, 'Oregon': 8, 'Memphis': 8,
    },
    2008: {
        'Kansas': 1, 'Memphis': 2,
        'North Carolina': 4, 'UCLA': 4,
        'Texas': 8, 'Louisville': 8, 'Stanford': 8, 'Davidson': 8,
    },
    2009: {
        'North Carolina': 1, 'Michigan St.': 2,
        'Villanova': 4, 'Connecticut': 4,
        'Oklahoma': 8, 'Louisville': 8, 'Syracuse': 8, 'Missouri': 8,
    },
    2010: {
        'Duke': 1, 'Butler': 2,
        'Michigan St.': 4, 'West Virginia': 4,
        'Kansas St.': 8, 'Kentucky': 8, 'Tennessee': 8, 'Baylor': 8,
    },
    2011: {
        'Connecticut': 1, 'Butler': 2,
        'VCU': 4, 'Kentucky': 4,
        'North Carolina': 8, 'Kansas': 8, 'Arizona': 8, 'Florida': 8,
    },
    2012: {
        'Kentucky': 1, 'Kansas': 2,
        'Louisville': 4, 'Ohio St.': 4,
        'Baylor': 8, 'North Carolina': 8, 'Syracuse': 8, 'Indiana': 8,
    },
    2013: {
        'Louisville': 1, 'Michigan': 2,
        'Syracuse': 4, 'Wichita St.': 4,
        'Duke': 8, 'Indiana': 8, 'Michigan St.': 8, 'Ohio St.': 8,
    },
    2014: {
        'Connecticut': 1, 'Kentucky': 2,
        'Florida': 4, 'Wisconsin': 4,
        'Michigan': 8, 'Michigan St.': 8, 'Arizona': 8, 'Iowa St.': 8,
    },
    2015: {
        'Duke': 1, 'Wisconsin': 2,
        'Michigan St.': 4, 'Kentucky': 4,
        'Arizona': 8, 'Notre Dame': 8, 'Gonzaga': 8, 'North Carolina': 8,
    },
    2016: {
        'Villanova': 1, 'North Carolina': 2,
        'Oklahoma': 4, 'Syracuse': 4,
        'Oregon': 8, 'Kansas': 8, 'Virginia': 8, 'Notre Dame': 8,
    },
    2017: {
        'North Carolina': 1, 'Gonzaga': 2,
        'South Carolina': 4, 'Oregon': 4,
        'Kansas': 8, 'Kentucky': 8, 'Florida': 8, 'Xavier': 8,
    },
    2018: {
        'Villanova': 1, 'Michigan': 2,
        'Kansas': 4, 'Loyola Chicago': 4,
        'Kentucky': 8, 'Texas Tech': 8, 'Duke': 8, 'Gonzaga': 8,
    },
    2019: {
        'Virginia': 1, 'Texas Tech': 2,
        'Michigan St.': 4, 'Auburn': 4,
        'Duke': 8, 'Gonzaga': 8, 'Purdue': 8, 'Kentucky': 8,
    },
    # 2020: No tournament (COVID)
    2021: {
        'Baylor': 1, 'Gonzaga': 2,
        'Houston': 4, 'UCLA': 4,
        'Michigan': 8, 'Arkansas': 8, 'Oregon St.': 8, 'USC': 8,
    },
    2022: {
        'Kansas': 1, 'North Carolina': 2,
        'Duke': 4, 'Villanova': 4,
        'Arkansas': 8, 'Miami FL': 8, 'Saint Peter\'s': 8, 'Houston': 8,
    },
    2023: {
        'Connecticut': 1, 'San Diego St.': 2,
        'Miami FL': 4, 'Florida Atlantic': 4,
        'Gonzaga': 8, 'Creighton': 8, 'Texas': 8, 'Kansas St.': 8,
    },
    2024: {
        'Connecticut': 1, 'Purdue': 2,
        'Alabama': 4, 'NC State': 4,
        'Duke': 8, 'Clemson': 8, 'Marquette': 8, 'Tennessee': 8,
    },
}


def add_round_data():
    """Add ROUND column to the unified dataset."""
    filepath = RAW_DIR / "KenPom Barttorvik.csv"
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        return

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} team-seasons")

    # Initialize ROUND column
    if 'ROUND' not in df.columns:
        df['ROUND'] = np.nan

    # For teams with a SEED but no ROUND, default to 64 (first-round loss)
    has_seed = df['SEED'].notna()
    df.loc[has_seed & df['ROUND'].isna(), 'ROUND'] = 64

    # Apply known tournament results
    updated = 0
    for year, results in TOURNAMENT_RESULTS.items():
        for team, round_val in results.items():
            mask = (df['YEAR'] == year) & (df['TEAM'] == team)
            if mask.any():
                df.loc[mask, 'ROUND'] = round_val
                updated += 1
            else:
                # Try fuzzy match
                year_teams = df[df['YEAR'] == year]['TEAM'].values
                for yt in year_teams:
                    if team.lower() in yt.lower() or yt.lower() in team.lower():
                        mask2 = (df['YEAR'] == year) & (df['TEAM'] == yt)
                        df.loc[mask2, 'ROUND'] = round_val
                        updated += 1
                        break

    # Save
    df.to_csv(filepath, index=False)

    # Report
    champions = df[df['ROUND'] == 1]
    print(f"\nUpdated {updated} tournament results")
    print(f"Champions found: {len(champions)}")
    print(f"\nChampions by year:")
    for _, row in champions.sort_values('YEAR').iterrows():
        print(f"  {int(row['YEAR'])}: {row['TEAM']} (Seed {int(row['SEED'])})")

    # Verify 1 champion per year
    champs_per_year = champions.groupby('YEAR').size()
    bad = champs_per_year[champs_per_year != 1]
    if len(bad) > 0:
        print(f"\nWARNING: Years with != 1 champion: {dict(bad)}")

    print(f"\nRound distribution (tournament teams only):")
    tourney = df[df['SEED'].notna()]
    print(tourney['ROUND'].value_counts().sort_index().to_string())


if __name__ == "__main__":
    add_round_data()
