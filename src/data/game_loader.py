"""
Game-level data loader for NCAA Tournament predictions.

This module handles:
- Loading tournament matchup data
- Pairing teams into actual games (each game has 2 rows in raw data)
- Creating train/test splits by year for game prediction
- Merging with team statistics for feature building

GAME STRUCTURE:
Raw data has one row per team per game. We pair them by:
- Same YEAR
- Same CURRENT ROUND  
- Same BY ROUND NO (game identifier within round)

The team with lower ROUND value (went further) won the game.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    TOURNAMENT_MATCHUPS_FILE,
    KENPOM_BARTTORVIK_FILE,
    MIN_YEAR,
    MAX_YEAR,
)


class GameLoader:
    """
    Loads tournament games as paired matchups.
    
    Converts raw data (one row per team per game) into 
    game-level data (one row per game with both teams).
    
    Attributes:
        games_df: DataFrame of all games [YEAR, TEAM_A, TEAM_B, WINNER, etc.]
        team_stats: DataFrame of team statistics
    """
    
    def __init__(self):
        """Initialize the game loader."""
        self.games_df: Optional[pd.DataFrame] = None
        self.team_stats: Optional[pd.DataFrame] = None
        self._raw_matchups: Optional[pd.DataFrame] = None
        
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all games and team statistics.
        
        Returns:
            Tuple of (games_df, team_stats_df)
        """
        self._load_matchups()
        self._load_team_stats()
        self._pair_games()
        
        return self.games_df, self.team_stats
    
    def _load_matchups(self) -> None:
        """Load raw tournament matchup data."""
        print(f"Loading Tournament Matchups from {TOURNAMENT_MATCHUPS_FILE}")
        
        self._raw_matchups = pd.read_csv(TOURNAMENT_MATCHUPS_FILE)
        
        # Filter to completed tournaments (have scores)
        self._raw_matchups = self._raw_matchups[
            self._raw_matchups['SCORE'].notna()
        ].copy()
        
        print(f"  Loaded {len(self._raw_matchups)} team-game entries")
        print(f"  Years: {sorted(self._raw_matchups['YEAR'].unique())}")
        
    def _load_team_stats(self) -> None:
        """Load KenPom/Barttorvik team statistics."""
        print(f"Loading Team Stats from {KENPOM_BARTTORVIK_FILE}")
        
        self.team_stats = pd.read_csv(KENPOM_BARTTORVIK_FILE)
        
        # Add SEED_NUM for numeric seed
        if self.team_stats['SEED'].dtype == 'object':
            self.team_stats['SEED_NUM'] = (
                self.team_stats['SEED']
                .astype(str)
                .str.extract(r'(\d+)')
                .astype(float)
            )
        else:
            self.team_stats['SEED_NUM'] = self.team_stats['SEED'].astype(float)
            
        # Filter to tournament teams
        self.team_stats = self.team_stats[
            self.team_stats['SEED'].notna()
        ].copy()
        
        print(f"  Loaded stats for {len(self.team_stats)} team-seasons")
        
    def _pair_games(self) -> None:
        """
        Pair raw matchup rows into games.
        
        Each game in raw data has 2 rows (one per team).
        We identify games by grouping on YEAR + CURRENT ROUND + game sequence.
        """
        print("Pairing matchups into games...")
        
        games_list = []
        df = self._raw_matchups.copy()
        
        # Sort to ensure consistent pairing
        df = df.sort_values(['YEAR', 'CURRENT ROUND', 'BY ROUND NO'])
        
        # Group games by year and round
        for year in df['YEAR'].unique():
            year_df = df[df['YEAR'] == year]
            
            for current_round in year_df['CURRENT ROUND'].unique():
                round_df = year_df[year_df['CURRENT ROUND'] == current_round]
                
                # Games are paired by adjacent rows (sorted by BY ROUND NO)
                # Each pair represents one game
                round_df = round_df.sort_values('BY ROUND NO', ascending=False)
                teams = round_df.to_dict('records')
                
                for i in range(0, len(teams), 2):
                    if i + 1 >= len(teams):
                        continue
                        
                    team1 = teams[i]
                    team2 = teams[i + 1]
                    
                    # Determine winner (lower ROUND = went further = won this game)
                    if team1['ROUND'] < team2['ROUND']:
                        winner = team1['TEAM']
                        loser = team2['TEAM']
                        winner_seed = team1['SEED']
                        loser_seed = team2['SEED']
                        winner_score = team1['SCORE']
                        loser_score = team2['SCORE']
                    elif team2['ROUND'] < team1['ROUND']:
                        winner = team2['TEAM']
                        loser = team1['TEAM']
                        winner_seed = team2['SEED']
                        loser_seed = team1['SEED']
                        winner_score = team2['SCORE']
                        loser_score = team1['SCORE']
                    else:
                        # Same ROUND means we need to use score to determine winner
                        if team1['SCORE'] > team2['SCORE']:
                            winner = team1['TEAM']
                            loser = team2['TEAM']
                            winner_seed = team1['SEED']
                            loser_seed = team2['SEED']
                            winner_score = team1['SCORE']
                            loser_score = team2['SCORE']
                        else:
                            winner = team2['TEAM']
                            loser = team1['TEAM']
                            winner_seed = team2['SEED']
                            loser_seed = team1['SEED']
                            winner_score = team2['SCORE']
                            loser_score = team1['SCORE']
                    
                    # Convert seeds to numeric
                    seed1 = self._parse_seed(team1['SEED'])
                    seed2 = self._parse_seed(team2['SEED'])
                    
                    # Store game with higher seed (lower number) as TEAM_A
                    if seed1 <= seed2:
                        game = {
                            'YEAR': year,
                            'ROUND': current_round,
                            'TEAM_A': team1['TEAM'],
                            'TEAM_B': team2['TEAM'],
                            'SEED_A': seed1,
                            'SEED_B': seed2,
                            'SCORE_A': team1['SCORE'],
                            'SCORE_B': team2['SCORE'],
                            'WINNER': winner,
                            'LOSER': loser,
                            'WINNER_SEED': self._parse_seed(winner_seed),
                            'LOSER_SEED': self._parse_seed(loser_seed),
                            'MARGIN': abs(team1['SCORE'] - team2['SCORE']),
                        }
                    else:
                        game = {
                            'YEAR': year,
                            'ROUND': current_round,
                            'TEAM_A': team2['TEAM'],
                            'TEAM_B': team1['TEAM'],
                            'SEED_A': seed2,
                            'SEED_B': seed1,
                            'SCORE_A': team2['SCORE'],
                            'SCORE_B': team1['SCORE'],
                            'WINNER': winner,
                            'LOSER': loser,
                            'WINNER_SEED': self._parse_seed(winner_seed),
                            'LOSER_SEED': self._parse_seed(loser_seed),
                            'MARGIN': abs(team1['SCORE'] - team2['SCORE']),
                        }
                    
                    games_list.append(game)
        
        self.games_df = pd.DataFrame(games_list)
        
        # Add upset flag (higher seed beat lower seed)
        self.games_df['IS_UPSET'] = (
            self.games_df['WINNER_SEED'] > self.games_df['LOSER_SEED']
        )
        
        print(f"  Created {len(self.games_df)} games")
        print(f"  Upsets: {self.games_df['IS_UPSET'].sum()} ({self.games_df['IS_UPSET'].mean():.1%})")
        
    def _parse_seed(self, seed) -> int:
        """Parse seed string to integer."""
        if pd.isna(seed):
            return 16
        if isinstance(seed, (int, float)):
            return int(seed)
        # Handle strings like "11a", "11b" (First Four)
        import re
        match = re.search(r'(\d+)', str(seed))
        return int(match.group(1)) if match else 16
    
    def get_train_test_split(
        self, 
        test_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test split for games.
        
        Train: all games from years < test_year
        Test: all games from test_year
        
        Args:
            test_year: Year to use as test set
            
        Returns:
            Tuple of (train_games, test_games)
        """
        if self.games_df is None:
            self.load_all()
            
        train = self.games_df[self.games_df['YEAR'] < test_year].copy()
        test = self.games_df[self.games_df['YEAR'] == test_year].copy()
        
        return train, test
    
    def get_years(self) -> List[int]:
        """Get list of available years."""
        if self.games_df is None:
            self.load_all()
        return sorted(self.games_df['YEAR'].unique())
    
    def get_games_by_round(self, year: int) -> pd.DataFrame:
        """Get games for a specific year, organized by round."""
        if self.games_df is None:
            self.load_all()
            
        year_games = self.games_df[self.games_df['YEAR'] == year].copy()
        return year_games.sort_values('ROUND', ascending=False)
    
    def describe(self) -> None:
        """Print summary statistics about the games data."""
        if self.games_df is None:
            self.load_all()
            
        print("\n" + "="*60)
        print("GAME DATA SUMMARY")
        print("="*60)
        
        print(f"\nTotal games: {len(self.games_df)}")
        print(f"Years: {self.get_years()}")
        
        print("\nGames per year:")
        print(self.games_df.groupby('YEAR').size())
        
        print("\nGames per round (sample year 2024):")
        y2024 = self.games_df[self.games_df['YEAR'] == 2024]
        print(y2024.groupby('ROUND').size().sort_index(ascending=False))
        
        print(f"\nTotal upsets: {self.games_df['IS_UPSET'].sum()}")
        print(f"Upset rate: {self.games_df['IS_UPSET'].mean():.1%}")
        
        print("\nUpset rate by round:")
        upset_by_round = self.games_df.groupby('ROUND')['IS_UPSET'].mean()
        print(upset_by_round.sort_index(ascending=False))


def main():
    """Test game loader."""
    loader = GameLoader()
    games, stats = loader.load_all()
    
    loader.describe()
    
    # Show sample games
    print("\n" + "="*60)
    print("SAMPLE GAMES (2024 Final Four + Championship)")
    print("="*60)
    
    final_games = games[(games['YEAR'] == 2024) & (games['ROUND'] <= 4)]
    print(final_games[['ROUND', 'TEAM_A', 'SEED_A', 'TEAM_B', 'SEED_B', 'WINNER', 'MARGIN']].to_string())
    
    # Test train/test split
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT (test year: 2024)")
    print("="*60)
    
    train, test = loader.get_train_test_split(2024)
    print(f"Train games: {len(train)} (years {train['YEAR'].min()}-{train['YEAR'].max()})")
    print(f"Test games: {len(test)} (year 2024)")


if __name__ == "__main__":
    main()
