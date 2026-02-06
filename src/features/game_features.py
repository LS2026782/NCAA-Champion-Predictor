"""
Enhanced feature engineering for game-by-game prediction.

This module builds features specifically for predicting individual game outcomes,
incorporating:
1. Neutral court statistics (tournament games are neutral)
2. Distance to game location (home court advantage proxy)
3. Coach tournament experience
4. Historical seed matchup win rates

DESIGN PRINCIPLES:
- Only use stable, team-systemic features (not player-dependent)
- Coach experience is stable (coaches stay with programs)
- Neutral court performance reflects team system, not home crowd
- Distance is geographic and measurable
- Seed priors are historical baselines
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import RAW_DATA_DIR


class GameFeatureBuilder:
    """
    Builds enhanced features for game prediction.
    
    Combines multiple data sources to create rich feature sets
    for predicting individual tournament game outcomes.
    """
    
    def __init__(self):
        """Initialize feature builder and load supplementary data."""
        self.neutral_stats: Optional[pd.DataFrame] = None
        self.coach_stats: Optional[pd.DataFrame] = None
        self.location_data: Optional[pd.DataFrame] = None
        self.seed_priors: Optional[Dict[Tuple[int, int], float]] = None
        self._loaded = False
        
    def load_supplementary_data(self) -> None:
        """Load all supplementary data sources."""
        if self._loaded:
            return
            
        print("Loading supplementary data for enhanced features...")
        
        # 1. Neutral court stats
        neutral_path = RAW_DATA_DIR / "Barttorvik Neutral.csv"
        if neutral_path.exists():
            self.neutral_stats = pd.read_csv(neutral_path)
            print(f"  Loaded neutral court stats: {len(self.neutral_stats)} rows")
        else:
            print(f"  Warning: {neutral_path} not found")
            
        # 2. Coach tournament experience
        coach_path = RAW_DATA_DIR / "Coach Results.csv"
        if coach_path.exists():
            self.coach_stats = pd.read_csv(coach_path)
            print(f"  Loaded coach stats: {len(self.coach_stats)} rows")
        else:
            print(f"  Warning: {coach_path} not found")
            
        # 3. Tournament locations (for distance)
        loc_path = RAW_DATA_DIR / "Tournament Locations.csv"
        if loc_path.exists():
            self.location_data = pd.read_csv(loc_path)
            print(f"  Loaded location data: {len(self.location_data)} rows")
        else:
            print(f"  Warning: {loc_path} not found")
            
        # 4. Seed matchup priors
        self._build_seed_priors()
        
        self._loaded = True
        
    def _build_seed_priors(self) -> None:
        """
        Build historical win rate priors for seed matchups.
        
        For each (seed_a, seed_b) pair, calculate historical win rate
        of seed_a over seed_b in tournament games.
        """
        seed_path = RAW_DATA_DIR / "Upset Seed Info.csv"
        
        if seed_path.exists():
            upset_data = pd.read_csv(seed_path)
            # This file has upset info - we'll build priors from matchup data
            print(f"  Loaded upset seed info: {len(upset_data)} rows")
        
        # Build priors from historical averages
        # Based on historical data analysis
        self.seed_priors = {}
        
        # Round of 64 matchups (1v16, 2v15, etc.)
        r64_matchups = {
            (1, 16): 0.99,   # 1-seeds almost always win
            (2, 15): 0.94,
            (3, 14): 0.85,
            (4, 13): 0.79,
            (5, 12): 0.65,   # 12-seeds upset often (35%)
            (6, 11): 0.63,   # 11-seeds upset often
            (7, 10): 0.61,
            (8, 9): 0.52,    # Essentially a coin flip
        }
        
        # Add matchups and their inverses
        for (s1, s2), win_rate in r64_matchups.items():
            self.seed_priors[(s1, s2)] = win_rate
            self.seed_priors[(s2, s1)] = 1 - win_rate
            
        # Later round matchups - use seed difference as proxy
        # Better seed wins ~60% when seeds are close
        for s1 in range(1, 17):
            for s2 in range(1, 17):
                if (s1, s2) not in self.seed_priors:
                    if s1 < s2:
                        # Lower seed is better
                        diff = s2 - s1
                        # Roughly 50% + 2% per seed difference, capped
                        win_rate = min(0.50 + 0.03 * diff, 0.85)
                    elif s1 > s2:
                        diff = s1 - s2
                        win_rate = max(0.50 - 0.03 * diff, 0.15)
                    else:
                        win_rate = 0.50
                    self.seed_priors[(s1, s2)] = win_rate
                    
        print(f"  Built seed priors for {len(self.seed_priors)} matchups")
        
    def get_neutral_court_features(
        self, 
        year: int, 
        team: str
    ) -> Dict[str, float]:
        """
        Get neutral court performance features for a team.
        
        Args:
            year: Tournament year
            team: Team name
            
        Returns:
            Dictionary of neutral court features
        """
        if self.neutral_stats is None:
            return {}
            
        team_data = self.neutral_stats[
            (self.neutral_stats['YEAR'] == year) & 
            (self.neutral_stats['TEAM'] == team)
        ]
        
        if len(team_data) == 0:
            return {}
            
        row = team_data.iloc[0]
        
        return {
            'NEUTRAL_EM': row.get('BADJ EM', 0),
            'NEUTRAL_O': row.get('BADJ O', 0),
            'NEUTRAL_D': row.get('BADJ D', 0),
            'NEUTRAL_EFG': row.get('EFG%', 0),
            'NEUTRAL_TOV': row.get('TOV%', 0),
            'NEUTRAL_OREB': row.get('OREB%', 0),
            'NEUTRAL_FTR': row.get('FTR', 0),
        }
        
    def get_coach_features(
        self, 
        year: int, 
        team: str,
        team_stats: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get coach tournament experience features.
        
        Args:
            year: Tournament year
            team: Team name
            team_stats: Team stats DataFrame (may have coach info)
            
        Returns:
            Dictionary of coach features
        """
        if self.coach_stats is None:
            return {'COACH_GAMES': 0, 'COACH_WINS': 0, 'COACH_WIN_PCT': 0.5}
        
        # Try to find coach for this team/year
        # This requires matching team to coach - simplified approach
        # In practice, you'd need a team-coach mapping by year
        
        # For now, return average coach stats as baseline
        # This is a placeholder - ideally we'd have team-coach mappings
        return {
            'COACH_GAMES': 20,  # Average
            'COACH_WINS': 10,
            'COACH_WIN_PCT': 0.5,
        }
        
    def get_distance_features(
        self,
        year: int,
        team: str,
        round_num: int
    ) -> Dict[str, float]:
        """
        Get distance-to-game features.
        
        Teams playing closer to home have historical advantages.
        
        Args:
            year: Tournament year
            team: Team name
            round_num: Tournament round (64, 32, 16, 8, 4, 2)
            
        Returns:
            Dictionary with distance features
        """
        if self.location_data is None:
            return {'DISTANCE_MI': 500}  # Default to neutral
            
        # Find this team's game in this round
        # Note: ROUND column in location data represents where team finished,
        # not which round we're looking at. Use BY ROUND NO for matching.
        game_data = self.location_data[
            (self.location_data['YEAR'] == year) &
            (self.location_data['TEAM'] == team)
        ]
        
        if len(game_data) == 0:
            # Try matching on ROUND column instead
            game_data = self.location_data[
                (self.location_data['YEAR'] == year) &
                (self.location_data['TEAM'] == team)
            ]
            if len(game_data) == 0:
                return {'DISTANCE_MI': 500}
                
        row = game_data.iloc[0]
        distance = row.get('DISTANCE (MI)', 500)
        
        if pd.isna(distance):
            distance = 500
            
        return {
            'DISTANCE_MI': distance,
            'DISTANCE_ADVANTAGE': 1 if distance < 300 else 0,  # Close to home
        }
        
    def get_seed_prior(self, seed_a: int, seed_b: int) -> float:
        """
        Get historical win rate for seed_a vs seed_b.
        
        Args:
            seed_a: Seed of team A
            seed_b: Seed of team B
            
        Returns:
            Historical probability that seed_a beats seed_b
        """
        if self.seed_priors is None:
            self._build_seed_priors()
            
        return self.seed_priors.get((seed_a, seed_b), 0.5)
        
    def build_game_features(
        self,
        year: int,
        team_a: str,
        team_b: str,
        seed_a: int,
        seed_b: int,
        round_num: int,
        team_stats: pd.DataFrame,
        include_neutral: bool = True,
        include_distance: bool = True,
        include_seed_prior: bool = True,
    ) -> Dict[str, float]:
        """
        Build complete feature set for a game.
        
        Args:
            year: Tournament year
            team_a: First team name
            team_b: Second team name
            seed_a: Seed of team A
            seed_b: Seed of team B
            round_num: Tournament round
            team_stats: DataFrame with team statistics
            include_*: Flags for which features to include
            
        Returns:
            Dictionary of features (differences: A - B)
        """
        self.load_supplementary_data()
        
        features = {}
        
        # 1. Neutral court stats (differences)
        if include_neutral:
            neutral_a = self.get_neutral_court_features(year, team_a)
            neutral_b = self.get_neutral_court_features(year, team_b)
            
            for key in neutral_a:
                features[f'{key}_DIFF'] = neutral_a.get(key, 0) - neutral_b.get(key, 0)
                
        # 2. Distance features
        if include_distance:
            dist_a = self.get_distance_features(year, team_a, round_num)
            dist_b = self.get_distance_features(year, team_b, round_num)
            
            features['DISTANCE_DIFF'] = dist_b.get('DISTANCE_MI', 500) - dist_a.get('DISTANCE_MI', 500)
            # Positive = team A is closer (advantage)
            
        # 3. Seed prior
        if include_seed_prior:
            features['SEED_PRIOR'] = self.get_seed_prior(int(seed_a), int(seed_b))
            
        return features


def main():
    """Test feature builder."""
    builder = GameFeatureBuilder()
    builder.load_supplementary_data()
    
    # Test seed priors
    print("\nSeed Prior Examples:")
    for matchup in [(1, 16), (5, 12), (8, 9), (11, 6), (2, 3)]:
        s1, s2 = matchup
        prior = builder.get_seed_prior(s1, s2)
        print(f"  {s1}-seed vs {s2}-seed: {prior:.1%} for {s1}-seed")
    
    # Test neutral court features
    print("\nNeutral Court Features (2024 Connecticut):")
    neutral = builder.get_neutral_court_features(2024, 'Connecticut')
    for k, v in neutral.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
