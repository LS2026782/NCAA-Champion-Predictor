"""
Bracket-Adjusted Feature Builder.

Adds post-Selection Sunday bracket factors:
1. POWER-PATH (favorable draw indicator) - HIGHLY SIGNIFICANT
2. PATH difficulty
3. Regional seed position

This model is for use AFTER the bracket is released!
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import RAW_DATA_DIR
from src.features.final_builder import FinalFeatureBuilder


class BracketAdjustedBuilder(FinalFeatureBuilder):
    """
    Extends FinalFeatureBuilder with bracket-specific features.
    Use this model AFTER Selection Sunday when bracket is known!
    """
    
    def __init__(self):
        super().__init__()
        
        # Add bracket features to the feature list
        self.feature_cols = self.feature_cols + [
            'POWER_PATH',      # Favorable draw indicator (HUGE predictor!)
            'PATH',            # Bracket path difficulty
            'BRACKET_EDGE',    # Combined bracket advantage
        ]
        
        # Load Heat Check data
        try:
            self.heatcheck_df = pd.read_csv(RAW_DATA_DIR / 'Heat Check Tournament Index.csv')
        except:
            self.heatcheck_df = None
    
    def build_features(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = False,
        all_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build features including bracket factors."""
        
        # First, build all the base features
        df = df.copy()
        df = self._add_seed_features(df)
        df = self._add_away_neutral_features(df)
        df = self._add_shooting_features(df)
        df = self._add_conference_features(df)
        df = self._add_momentum_features(df)
        df = self._add_resume_features(df)
        df = self._add_composite_features(df)
        
        # Add bracket-specific features
        df = self._add_bracket_features(df)
        
        # Get available features
        available = [f for f in self.feature_cols if f in df.columns]
        missing = [f for f in self.feature_cols if f not in df.columns]
        
        if missing:
            print(f"  Missing features (filled with median): {missing}")
            for f in missing:
                df[f] = 0
        
        X = df[self.feature_cols].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().any():
                median = X[col].median()
                X[col] = X[col].fillna(median if pd.notna(median) else 0)
        
        # Scale
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self._fitted = True
        elif self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        self.feature_names = self.feature_cols
        return df, X_scaled
    
    def _add_bracket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bracket-specific features from Heat Check data."""
        
        if self.heatcheck_df is None:
            df['POWER_PATH'] = 0
            df['PATH'] = 70  # avg
            df['BRACKET_EDGE'] = 0
            return df
        
        # Merge Heat Check data
        hc_cols = ['YEAR', 'TEAM', 'POWER', 'PATH', 'POWER-PATH']
        hc_subset = self.heatcheck_df[hc_cols].copy()
        hc_subset.columns = ['YEAR', 'TEAM', 'HC_POWER', 'PATH', 'POWER_PATH']
        
        df = df.merge(hc_subset, on=['YEAR', 'TEAM'], how='left')
        
        # Fill missing with neutral values
        df['POWER_PATH'] = df['POWER_PATH'].fillna(0)
        df['PATH'] = df['PATH'].fillna(df['PATH'].median() if df['PATH'].notna().any() else 70)
        
        # Create bracket edge composite
        # Normalize POWER_PATH (typically ranges from -50 to +30)
        pp_min = df['POWER_PATH'].min()
        pp_max = df['POWER_PATH'].max()
        if pp_max > pp_min:
            df['BRACKET_EDGE'] = (df['POWER_PATH'] - pp_min) / (pp_max - pp_min)
        else:
            df['BRACKET_EDGE'] = 0.5
        
        return df


def backtest_bracket_model():
    """Backtest the bracket-adjusted model."""
    import sys
    sys.path.insert(0, '.')
    
    from src.data.loader import DataLoader
    from src.models.champion_model import ChampionPredictor
    from config.settings import FIRST_TEST_YEAR
    
    print("="*70)
    print("BRACKET-ADJUSTED MODEL BACKTEST")
    print("="*70)
    print("This model includes POWER-PATH (favorable draw indicator)")
    print()
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        builder = BracketAdjustedBuilder()
        train_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
        test_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
        
        y_train = train_feat['IS_CHAMPION'].values
        
        model = ChampionPredictor(model_type='logreg')
        model.fit(X_train, y_train, feature_names=builder.get_feature_names())
        
        probs = model.predict_proba(X_test)
        test_feat = test_feat.copy()
        test_feat['PROB'] = probs
        test_feat['RANK'] = test_feat['PROB'].rank(ascending=False).astype(int)
        
        champ = test_feat[test_feat['IS_CHAMPION'] == 1].iloc[0]
        
        pp = champ.get('POWER_PATH', 0)
        results.append({
            'year': test_year,
            'champion': champ['TEAM'],
            'seed': int(champ['SEED']),
            'rank': int(champ['RANK']),
            'power_path': pp
        })
        
        in_top5 = "Yes" if champ['RANK'] <= 5 else "No"
        print(f"{test_year}: {champ['TEAM']:<20} Seed={int(champ['SEED'])} "
              f"Rank={int(champ['RANK']):2d} PP={pp:+.1f} Top5={in_top5}")
    
    # Summary
    ranks = [r['rank'] for r in results]
    
    print("\n" + "="*70)
    print("BRACKET-ADJUSTED MODEL SUMMARY")
    print("="*70)
    print(f"Mean Champion Rank:   {np.mean(ranks):.2f}")
    print(f"Median Champion Rank: {np.median(ranks):.1f}")
    print(f"Top-1 Rate:  {sum(r==1 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-3 Rate:  {sum(r<=3 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-5 Rate:  {sum(r<=5 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-10 Rate: {sum(r<=10 for r in ranks)/len(ranks)*100:.1f}%")
    
    return results


if __name__ == "__main__":
    backtest_bracket_model()
