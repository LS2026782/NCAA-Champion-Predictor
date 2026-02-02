"""
Ultimate Feature Builder - Combining ALL significant features discovered.

Features included:
1. Core Efficiency: KADJ EM, KADJ O, KADJ D, BARTHAG
2. Four Factors: EFG%, TOV%, OREB%, etc.
3. Momentum: Preseason improvement, Q1 wins
4. NEW: WAB (Wins Above Bubble) - MOST PREDICTIVE
5. NEW: FT% (Clutch free throws)
6. NEW: BLK% (Rim protection)
7. NEW: EFF HGT (Height advantage)
8. Seed-based features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import RAW_DATA_DIR


class UltimateFeatureBuilder:
    """
    Ultimate feature builder combining all significant predictors.
    """
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False
        self._load_auxiliary_data()
        
        # ULTIMATE FEATURE SET
        self.feature_cols = [
            # Core Efficiency (proven predictive)
            'KADJ EM',
            'KADJ O', 
            'KADJ D',
            'BARTHAG',
            
            # Four Factors - Offense
            'EFG%',
            'TOV%',
            'OREB%',
            'FTR',
            
            # Four Factors - Defense
            'EFG%D',
            'TOV%D',
            'DREB%',
            'FTRD',
            
            # NEW: Highly significant unexplored features
            'WAB',        # Wins Above Bubble - HUGE predictor
            'FT%',        # Free throw clutch factor
            'BLK%',       # Rim protection
            'EFF HGT',    # Height advantage
            'TALENT',     # Recruiting
            
            # Schedule & Seed
            'ELITE SOS',
            'SEED_STRENGTH',
            
            # Momentum features
            'MOMENTUM_EM',
            'Q1_WINS',
            'Q1_Q2_WINS',
            
            # Composite features
            'EM_X_SOS',
            'CLUTCH_COMPOSITE',  # NEW: FT% + close game performance
        ]
        
    def _load_auxiliary_data(self):
        """Load preseason and resume data."""
        try:
            self.preseason_df = pd.read_csv(RAW_DATA_DIR / 'KenPom Preseason.csv')
        except:
            self.preseason_df = None
            
        try:
            self.resume_df = pd.read_csv(RAW_DATA_DIR / 'Resumes.csv')
        except:
            self.resume_df = None
    
    def build_features(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = False,
        all_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build ultimate feature matrix."""
        df = df.copy()
        
        # Add seed features
        df = self._add_seed_features(df)
        
        # Add momentum features
        df = self._add_momentum_features(df)
        
        # Add resume features (Q1 wins)
        df = self._add_resume_features(df)
        
        # Add composite features
        df = self._add_composite_features(df)
        
        # Get available features
        available = [f for f in self.feature_cols if f in df.columns]
        missing = [f for f in self.feature_cols if f not in df.columns]
        
        if missing:
            print(f"  Missing features (will be filled with 0): {missing}")
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
    
    def _add_seed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seed-based features."""
        if df['SEED'].dtype == 'object':
            df['SEED_NUM'] = df['SEED'].astype(str).str.extract(r'(\d+)').astype(float)
        else:
            df['SEED_NUM'] = df['SEED'].astype(float)
        
        df['SEED_STRENGTH'] = 17 - df['SEED_NUM']
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum from preseason data."""
        if self.preseason_df is None:
            df['MOMENTUM_EM'] = 0
            return df
        
        preseason_cols = ['YEAR', 'TEAM', 'KADJ EM CHANGE']
        preseason_subset = self.preseason_df[preseason_cols].copy()
        preseason_subset.columns = ['YEAR', 'TEAM', 'MOMENTUM_EM']
        
        df = df.merge(preseason_subset, on=['YEAR', 'TEAM'], how='left')
        df['MOMENTUM_EM'] = df['MOMENTUM_EM'].fillna(0)
        
        return df
    
    def _add_resume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add resume features (Q1 wins, etc.)."""
        if self.resume_df is None:
            df['Q1_WINS'] = 0
            df['Q1_Q2_WINS'] = 0
            return df
        
        resume_cols = ['YEAR', 'TEAM', 'Q1 W', 'Q1 PLUS Q2 W']
        resume_subset = self.resume_df[resume_cols].copy()
        resume_subset.columns = ['YEAR', 'TEAM', 'Q1_WINS', 'Q1_Q2_WINS']
        
        df = df.merge(resume_subset, on=['YEAR', 'TEAM'], how='left')
        df['Q1_WINS'] = df['Q1_WINS'].fillna(df['Q1_WINS'].median())
        df['Q1_Q2_WINS'] = df['Q1_Q2_WINS'].fillna(df['Q1_Q2_WINS'].median())
        
        return df
    
    def _add_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite features."""
        
        # EM x SOS interaction
        if 'KADJ EM' in df.columns and 'ELITE SOS' in df.columns:
            sos_min = df['ELITE SOS'].min()
            sos_max = df['ELITE SOS'].max()
            sos_norm = (df['ELITE SOS'] - sos_min) / (sos_max - sos_min + 1e-8)
            df['EM_X_SOS'] = df['KADJ EM'] * (0.5 + sos_norm)
        else:
            df['EM_X_SOS'] = 0
        
        # Clutch composite: FT% + WAB (both indicate ability to win close games)
        if 'FT%' in df.columns and 'WAB' in df.columns:
            ft_norm = (df['FT%'] - df['FT%'].min()) / (df['FT%'].max() - df['FT%'].min() + 1e-8)
            wab_norm = (df['WAB'] - df['WAB'].min()) / (df['WAB'].max() - df['WAB'].min() + 1e-8)
            df['CLUTCH_COMPOSITE'] = 0.5 * ft_norm + 0.5 * wab_norm
        else:
            df['CLUTCH_COMPOSITE'] = 0
            
        return df
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names if hasattr(self, 'feature_names') else []


def backtest_ultimate():
    """Backtest the ultimate model."""
    import sys
    sys.path.insert(0, '.')
    
    from src.data.loader import DataLoader
    from src.models.champion_model import ChampionPredictor
    from config.settings import FIRST_TEST_YEAR
    
    print("="*70)
    print("ULTIMATE MODEL BACKTEST")
    print("="*70)
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        builder = UltimateFeatureBuilder()
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
        
        results.append({
            'year': test_year,
            'champion': champ['TEAM'],
            'seed': int(champ['SEED']),
            'rank': int(champ['RANK']),
            'prob': champ['PROB']
        })
        
        in_top5 = "Yes" if champ['RANK'] <= 5 else "No"
        print(f"{test_year}: {champ['TEAM']:<20} Seed={int(champ['SEED'])} "
              f"Rank={int(champ['RANK']):2d} Prob={champ['PROB']:.3f} Top5={in_top5}")
    
    # Summary
    ranks = [r['rank'] for r in results]
    
    print("\n" + "="*70)
    print("ULTIMATE MODEL SUMMARY")
    print("="*70)
    print(f"Mean Champion Rank:   {np.mean(ranks):.2f}")
    print(f"Median Champion Rank: {np.median(ranks):.1f}")
    print(f"Top-1 Rate:  {sum(r==1 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-5 Rate:  {sum(r<=5 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-10 Rate: {sum(r<=10 for r in ranks)/len(ranks)*100:.1f}%")
    
    return results


if __name__ == "__main__":
    backtest_ultimate()
