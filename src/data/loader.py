"""
Data loading and preprocessing for NCAA Championship Prediction.

This module handles:
- Loading raw CSV files from Kaggle datasets
- Merging KenPom/Barttorvik metrics with tournament results
- Creating champion labels from tournament progression
- Filtering to tournament teams only (to prevent leakage)

CRITICAL LEAKAGE PREVENTION:
- Only uses pre-tournament team statistics
- Tournament results used ONLY for labels, never as features
- Strict temporal splits enforced in backtesting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    KENPOM_BARTTORVIK_FILE,
    TOURNAMENT_MATCHUPS_FILE,
    TEAM_RESULTS_FILE,
    MIN_YEAR,
    MAX_YEAR,
    ROUND_CHAMPION,
)


class DataLoader:
    """
    Loads and preprocesses NCAA tournament data for champion prediction.
    
    Responsibilities:
    - Load raw data files
    - Merge team stats with tournament outcomes
    - Create binary champion labels
    - Filter to valid year range
    
    Attributes:
        kenpom_df: KenPom/Barttorvik team statistics
        matchups_df: Tournament matchup results
        team_results_df: Historical team performance
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Optional override for data directory path
        """
        self.kenpom_df: Optional[pd.DataFrame] = None
        self.matchups_df: Optional[pd.DataFrame] = None
        self.team_results_df: Optional[pd.DataFrame] = None
        self._merged_df: Optional[pd.DataFrame] = None
        
    def load_all(self) -> pd.DataFrame:
        """
        Load all data sources and return merged DataFrame.
        
        Returns:
            DataFrame with team stats and tournament outcomes merged
        """
        self._load_kenpom_barttorvik()
        self._load_tournament_matchups()
        self._create_labels()
        return self._merged_df
    
    def _load_kenpom_barttorvik(self) -> None:
        """
        Load KenPom/Barttorvik efficiency metrics.
        
        This file contains pre-tournament team statistics including:
        - Adjusted offensive/defensive efficiency
        - Four factors (eFG%, TO%, ORB%, FTR)
        - Experience and talent metrics
        - Strength of schedule
        """
        print(f"Loading KenPom/Barttorvik data from {KENPOM_BARTTORVIK_FILE}")
        
        self.kenpom_df = pd.read_csv(KENPOM_BARTTORVIK_FILE)
        
        # Filter to valid year range
        self.kenpom_df = self.kenpom_df[
            (self.kenpom_df['YEAR'] >= MIN_YEAR) & 
            (self.kenpom_df['YEAR'] <= MAX_YEAR)
        ].copy()
        
        # Only keep tournament teams (have a seed)
        # This is key: we only predict on tournament teams
        self.kenpom_df = self.kenpom_df[
            self.kenpom_df['SEED'].notna()
        ].copy()
        
        print(f"  Loaded {len(self.kenpom_df)} team-seasons ({MIN_YEAR}-{MAX_YEAR})")
        print(f"  Years: {sorted(self.kenpom_df['YEAR'].unique())}")
        
    def _load_tournament_matchups(self) -> None:
        """
        Load tournament matchup data for determining outcomes.
        
        Used to identify:
        - Which teams won the championship
        - How far each team progressed
        """
        print(f"Loading Tournament Matchups from {TOURNAMENT_MATCHUPS_FILE}")
        
        self.matchups_df = pd.read_csv(TOURNAMENT_MATCHUPS_FILE)
        
        # Filter to valid year range
        self.matchups_df = self.matchups_df[
            (self.matchups_df['YEAR'] >= MIN_YEAR) & 
            (self.matchups_df['YEAR'] <= MAX_YEAR)
        ].copy()
        
        print(f"  Loaded {len(self.matchups_df)} tournament entries")
        
    def _create_labels(self) -> None:
        """
        Create champion labels from the ROUND column in KenPom data.
        
        The ROUND column in KenPom/Barttorvik represents "teams remaining":
        - 1 = Champion (won it all)
        - 2 = Runner-up (lost in championship)
        - 4 = Final Four (lost in semifinal)
        - 8 = Elite Eight
        - 16 = Sweet Sixteen
        - 32 = Round of 32
        - 64 = Round of 64 (first round loss)
        - 68 = First Four (play-in game)
        
        Lower number = further in tournament = better performance.
        """
        print("Creating champion labels...")
        
        # Use the existing ROUND column from KenPom data
        self._merged_df = self.kenpom_df.copy()
        
        # Fill NaN rounds (shouldn't happen for tournament teams)
        self._merged_df['ROUND'] = self._merged_df['ROUND'].fillna(68)
        
        # Create binary champion label (ROUND=1 means champion)
        self._merged_df['IS_CHAMPION'] = (
            self._merged_df['ROUND'] == 1
        ).astype(int)
        
        # Create Final Four label (ROUND <= 4 means made Final Four)
        self._merged_df['MADE_FINAL_FOUR'] = (
            self._merged_df['ROUND'] <= 4
        ).astype(int)
        
        # Create Elite Eight label (ROUND <= 8)
        self._merged_df['MADE_ELITE_EIGHT'] = (
            self._merged_df['ROUND'] <= 8
        ).astype(int)
        
        # Verify champion counts
        champs_per_year = self._merged_df.groupby('YEAR')['IS_CHAMPION'].sum()
        print(f"  Champions per year: {dict(champs_per_year)}")
        print(f"  Total champions: {self._merged_df['IS_CHAMPION'].sum()}")
        print(f"  Total teams: {len(self._merged_df)}")
        
        # Sanity check: should have exactly 1 champion per year
        if not all(champs_per_year == 1):
            bad_years = champs_per_year[champs_per_year != 1]
            print(f"  WARNING: Unexpected champion counts in years: {dict(bad_years)}")
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the merged and labeled DataFrame.
        
        Returns:
            DataFrame with all features and labels
        """
        if self._merged_df is None:
            self.load_all()
        return self._merged_df.copy()
    
    def get_train_test_split(
        self, 
        test_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test split for a given test year.
        
        Implements strict temporal splitting to prevent leakage:
        - Train: all years < test_year
        - Test: only test_year
        
        Args:
            test_year: Year to use as test set
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self._merged_df is None:
            self.load_all()
            
        train_df = self._merged_df[self._merged_df['YEAR'] < test_year].copy()
        test_df = self._merged_df[self._merged_df['YEAR'] == test_year].copy()
        
        return train_df, test_df
    
    def get_years(self) -> list:
        """Get list of available years in the data."""
        if self._merged_df is None:
            self.load_all()
        return sorted(self._merged_df['YEAR'].unique())
    
    def describe_class_balance(self) -> pd.DataFrame:
        """
        Describe the class balance in the dataset.
        
        Returns:
            DataFrame with class distribution statistics
        """
        if self._merged_df is None:
            self.load_all()
            
        stats = {
            'Total Teams': len(self._merged_df),
            'Champions': self._merged_df['IS_CHAMPION'].sum(),
            'Non-Champions': (1 - self._merged_df['IS_CHAMPION']).sum(),
            'Champion Rate': self._merged_df['IS_CHAMPION'].mean(),
            'Imbalance Ratio': (1 - self._merged_df['IS_CHAMPION']).sum() / 
                              max(self._merged_df['IS_CHAMPION'].sum(), 1),
        }
        
        return pd.DataFrame([stats])


def main():
    """Test data loading functionality."""
    loader = DataLoader()
    df = loader.load_all()
    
    print("\n" + "="*60)
    print("DATA LOADING SUMMARY")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"\nYears: {loader.get_years()}")
    
    print("\nClass Balance:")
    print(loader.describe_class_balance().to_string())
    
    print("\nSample of champion teams:")
    champs = df[df['IS_CHAMPION'] == 1][['YEAR', 'TEAM', 'SEED', 'KADJ EM', 'BARTHAG']]
    print(champs.to_string())
    
    # Test train/test split
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT TEST (Year 2024)")
    print("="*60)
    
    train_df, test_df = loader.get_train_test_split(2024)
    print(f"Train shape: {train_df.shape} (years {train_df['YEAR'].min()}-{train_df['YEAR'].max()})")
    print(f"Test shape: {test_df.shape} (year {test_df['YEAR'].unique()})")
    print(f"Train champions: {train_df['IS_CHAMPION'].sum()}")
    print(f"Test champion: {test_df[test_df['IS_CHAMPION']==1]['TEAM'].values}")
    

if __name__ == "__main__":
    main()
