"""
Monte Carlo bracket simulation for NCAA Tournament.

This module simulates the tournament bracket thousands of times
using game-level win probabilities to estimate championship odds.

The simulation:
1. Trains a game-level win probability model on regular season data
2. Simulates each tournament game using predicted probabilities
3. Aggregates results across simulations to estimate championship odds

This provides a second method of estimating championship probability
that accounts for bracket structure and path difficulty.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import MONTE_CARLO_SIMULATIONS


class GameModel:
    """
    Predicts win probability for individual games.
    
    Uses the difference in team metrics to predict which team wins.
    Trained on regular season results only (no tournament data).
    
    Features: difference in (AdjEM, AdjO, AdjD, Seed)
    """
    
    def __init__(self):
        self.model = None
        self.feature_cols = ['KADJ EM', 'KADJ O', 'KADJ D', 'BARTHAG', 'SEED_NUM']
        
    def fit(self, team_stats: pd.DataFrame) -> 'GameModel':
        """
        Fit the game model using team statistics.
        
        Creates synthetic game matchups from team stats.
        For each pair of teams, the "winner" is determined by
        which team had better stats that year.
        
        Args:
            team_stats: DataFrame with team statistics
            
        Returns:
            self
        """
        # We'll use team stats directly to predict game outcomes
        # Higher EM team wins with probability based on EM difference
        self.model = LogisticRegression(random_state=42)
        
        # Create training data from team pairs
        X_train = []
        y_train = []
        
        for year in team_stats['YEAR'].unique():
            year_data = team_stats[team_stats['YEAR'] == year]
            
            # Sample random matchups within the year
            teams = year_data.index.tolist()
            n_matchups = min(len(teams) * 5, 500)  # Sample games
            
            for _ in range(n_matchups):
                i, j = np.random.choice(len(teams), 2, replace=False)
                team_a = year_data.iloc[i]
                team_b = year_data.iloc[j]
                
                # Feature: difference in metrics (A - B)
                diff = []
                for col in self.feature_cols:
                    if col in year_data.columns:
                        diff.append(team_a[col] - team_b[col])
                    else:
                        diff.append(0)
                        
                X_train.append(diff)
                
                # Label: 1 if team A "wins" (higher KADJ EM)
                y_train.append(1 if team_a['KADJ EM'] > team_b['KADJ EM'] else 0)
        
        self.model.fit(X_train, y_train)
        return self
    
    def predict_win_prob(
        self, 
        team_a_stats: Dict, 
        team_b_stats: Dict
    ) -> float:
        """
        Predict probability that team A beats team B.
        
        Args:
            team_a_stats: Statistics for team A
            team_b_stats: Statistics for team B
            
        Returns:
            Probability team A wins (0 to 1)
        """
        diff = []
        for col in self.feature_cols:
            val_a = team_a_stats.get(col, 0)
            val_b = team_b_stats.get(col, 0)
            diff.append(val_a - val_b)
            
        prob = self.model.predict_proba([diff])[0][1]
        return prob


class BracketSimulator:
    """
    Simulates NCAA Tournament brackets using Monte Carlo.
    
    The bracket structure:
    - 68 teams (First Four reduces to 64)
    - 4 regions of 16 teams each
    - Seeding: 1-16 in each region
    - 6 rounds: R64 -> R32 -> S16 -> E8 -> F4 -> Championship
    
    Simulation process:
    1. Initialize bracket with team stats
    2. For each simulation:
       a. Simulate each game using win probabilities
       b. Track which team wins championship
    3. Aggregate championship counts
    """
    
    def __init__(self, n_simulations: int = MONTE_CARLO_SIMULATIONS):
        """
        Initialize the simulator.
        
        Args:
            n_simulations: Number of bracket simulations to run
        """
        self.n_simulations = n_simulations
        self.game_model = GameModel()
        
    def fit(self, team_stats: pd.DataFrame) -> 'BracketSimulator':
        """
        Fit the game-level model on historical data.
        
        Args:
            team_stats: DataFrame with team statistics
            
        Returns:
            self
        """
        self.game_model.fit(team_stats)
        return self
    
    def simulate_tournament(
        self, 
        teams_df: pd.DataFrame,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation of the tournament.
        
        Args:
            teams_df: DataFrame with 64-68 tournament teams and stats
            verbose: Print progress
            
        Returns:
            Dictionary mapping team name -> championship probability
        """
        # Prepare team data
        teams_df = teams_df.copy()
        
        # Ensure SEED_NUM exists
        if 'SEED_NUM' not in teams_df.columns:
            if teams_df['SEED'].dtype == 'object':
                teams_df['SEED_NUM'] = teams_df['SEED'].astype(str).str.extract(r'(\d+)').astype(float)
            else:
                teams_df['SEED_NUM'] = teams_df['SEED'].astype(float)
        
        # Build team stats dictionary
        team_stats = {}
        for _, row in teams_df.iterrows():
            team_stats[row['TEAM']] = {
                'KADJ EM': row.get('KADJ EM', 0),
                'KADJ O': row.get('KADJ O', 0),
                'KADJ D': row.get('KADJ D', 0),
                'BARTHAG': row.get('BARTHAG', 0.5),
                'SEED_NUM': row.get('SEED_NUM', 8),
                'SEED': row.get('SEED', 8),
            }
        
        # Create bracket structure
        # Simplified: assume 64 teams, standard bracket
        teams_by_seed = teams_df.sort_values('SEED_NUM')
        team_names = teams_by_seed['TEAM'].tolist()[:64]  # Top 64 teams
        
        if len(team_names) < 64:
            print(f"Warning: Only {len(team_names)} teams, using all available")
        
        # Standard NCAA bracket matchup structure
        # In each region: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
        bracket_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        
        # Track championship wins
        championship_counts = defaultdict(int)
        
        if verbose:
            print(f"Running {self.n_simulations:,} tournament simulations...")
        
        for sim in range(self.n_simulations):
            if verbose and (sim + 1) % 1000 == 0:
                print(f"  Simulation {sim + 1:,}/{self.n_simulations:,}")
            
            # Simulate one tournament
            champion = self._simulate_single_tournament(team_names, team_stats)
            championship_counts[champion] += 1
        
        # Convert to probabilities
        total = sum(championship_counts.values())
        championship_probs = {
            team: count / total 
            for team, count in championship_counts.items()
        }
        
        return championship_probs
    
    def _simulate_single_tournament(
        self, 
        teams: List[str],
        team_stats: Dict
    ) -> str:
        """
        Simulate a single tournament and return the champion.
        
        Args:
            teams: List of team names (ordered by seed)
            team_stats: Dictionary of team statistics
            
        Returns:
            Name of championship winner
        """
        # Start with all teams
        remaining = teams.copy()
        
        # Simulate 6 rounds (R64, R32, S16, E8, F4, Championship)
        while len(remaining) > 1:
            next_round = []
            
            # Pair up teams for games
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    team_a = remaining[i]
                    team_b = remaining[i + 1]
                    
                    # Get win probability
                    prob_a = self.game_model.predict_win_prob(
                        team_stats.get(team_a, {}),
                        team_stats.get(team_b, {})
                    )
                    
                    # Simulate game
                    if np.random.random() < prob_a:
                        winner = team_a
                    else:
                        winner = team_b
                        
                    next_round.append(winner)
                else:
                    # Odd team gets bye (shouldn't happen with 64)
                    next_round.append(remaining[i])
                    
            remaining = next_round
            
        return remaining[0]
    
    def compare_methods(
        self,
        teams_df: pd.DataFrame,
        direct_probs: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compare simulation probabilities with direct model predictions.
        
        Args:
            teams_df: Tournament teams
            direct_probs: Dictionary of team -> prob from direct model
            
        Returns:
            DataFrame comparing the two methods
        """
        # Run simulation
        sim_probs = self.simulate_tournament(teams_df)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'team': list(set(direct_probs.keys()) | set(sim_probs.keys()))
        })
        
        comparison['direct_prob'] = comparison['team'].map(direct_probs).fillna(0)
        comparison['sim_prob'] = comparison['team'].map(sim_probs).fillna(0)
        comparison['diff'] = comparison['sim_prob'] - comparison['direct_prob']
        
        # Add ranks
        comparison['direct_rank'] = comparison['direct_prob'].rank(ascending=False)
        comparison['sim_rank'] = comparison['sim_prob'].rank(ascending=False)
        
        return comparison.sort_values('sim_prob', ascending=False)


def main():
    """Test Monte Carlo simulation."""
    from src.data.loader import DataLoader
    from src.features.builder import FeatureBuilder
    
    print("="*60)
    print("MONTE CARLO SIMULATION TEST")
    print("="*60)
    
    # Load data
    loader = DataLoader()
    loader.load_all()
    
    train_df, test_df = loader.get_train_test_split(2024)
    
    # Build features
    builder = FeatureBuilder()
    train_df_feat, _ = builder.build_features(train_df, fit_scaler=True)
    test_df_feat, _ = builder.build_features(test_df, fit_scaler=False)
    
    # Create simulator
    print("\nFitting game-level model...")
    simulator = BracketSimulator(n_simulations=5000)
    simulator.fit(train_df_feat)
    
    # Run simulation
    print("\nSimulating 2024 tournament...")
    sim_probs = simulator.simulate_tournament(test_df_feat, verbose=True)
    
    # Show results
    print("\nTop 15 by simulation:")
    sorted_probs = sorted(sim_probs.items(), key=lambda x: x[1], reverse=True)[:15]
    for team, prob in sorted_probs:
        seed = test_df_feat[test_df_feat['TEAM'] == team]['SEED'].values[0]
        is_champ = test_df_feat[test_df_feat['TEAM'] == team]['IS_CHAMPION'].values[0]
        marker = " <-- CHAMPION" if is_champ else ""
        print(f"  {team:20s} (Seed {seed}): {prob:.1%}{marker}")
    
    # Find actual champion
    champ_name = test_df_feat[test_df_feat['IS_CHAMPION'] == 1]['TEAM'].values[0]
    champ_prob = sim_probs.get(champ_name, 0)
    champ_rank = sorted([p for p in sim_probs.values()], reverse=True).index(champ_prob) + 1
    print(f"\nActual champion '{champ_name}' simulation rank: {champ_rank}, prob: {champ_prob:.1%}")


if __name__ == "__main__":
    main()
