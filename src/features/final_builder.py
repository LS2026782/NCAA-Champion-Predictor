"""
FINAL Feature Builder - Everything we've discovered.

Key Features:
1. Away-Neutral EM (tournaments are neutral court!) - HUGE
2. Dunk Share (rim pressure, easy baskets) - HUGE  
3. Conference historical success
4. WAB (Wins Above Bubble)
5. FT% (Clutch free throws)
6. BLK% (Rim protection)
7. Core efficiency metrics
8. Momentum features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import RAW_DATA_DIR


class FinalFeatureBuilder:
    """
    Final feature builder with ALL discovered predictive features.
    """
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False
        self._load_auxiliary_data()
        
        # FINAL FEATURE SET - Everything that matters
        self.feature_cols = [
            # Core Efficiency
            'KADJ EM',
            'KADJ O', 
            'KADJ D',
            'BARTHAG',
            
            # NEW: Away-Neutral Performance (HUGE!)
            'AWAY_NEUTRAL_EM',
            'HOME_AWAY_DROP',  # How much they drop away from home
            
            # NEW: Shot Quality (HUGE!)
            'DUNKS_SHARE',     # Easy baskets
            'CLOSE_TWOS_SHARE',
            
            # Four Factors
            'EFG%',
            'TOV%',
            'OREB%',
            'FTR',
            'EFG%D',
            'TOV%D',
            'DREB%',
            
            # Proven significant features
            'WAB',        # Wins Above Bubble
            'FT%',        # Clutch free throws
            'BLK%',       # Rim protection
            'EFF HGT',    # Height
            'TALENT',     # Recruiting
            
            # Schedule & Seed
            'ELITE SOS',
            'SEED_STRENGTH',
            
            # NEW: Conference historical success
            'CONF_CHAMP_RATE',
            
            # Momentum
            'MOMENTUM_EM',
            'Q1_WINS',
            
            # Composites
            'EM_X_SOS',
            'CLUTCH_COMPOSITE',
            'RIM_DOMINANCE',  # NEW: Dunks + BLK% combined
        ]
        
    def _load_auxiliary_data(self):
        """Load all auxiliary data sources."""
        try:
            self.preseason_df = pd.read_csv(RAW_DATA_DIR / 'KenPom Preseason.csv')
        except:
            self.preseason_df = None
            
        try:
            self.resume_df = pd.read_csv(RAW_DATA_DIR / 'Resumes.csv')
        except:
            self.resume_df = None
            
        try:
            self.away_df = pd.read_csv(RAW_DATA_DIR / 'Barttorvik Away-Neutral.csv')
        except:
            self.away_df = None
            
        try:
            self.shots_df = pd.read_csv(RAW_DATA_DIR / 'Shooting Splits.csv')
        except:
            self.shots_df = None
            
        try:
            self.conf_df = pd.read_csv(RAW_DATA_DIR / 'Conference Results.csv')
            # Convert CHAMP% to numeric
            self.conf_df['CHAMP_RATE'] = self.conf_df['CHAMP%'].apply(
                lambda x: float(str(x).replace('%', '')) / 100 if pd.notna(x) else 0
            )
        except:
            self.conf_df = None
    
    def build_features(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = False,
        all_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build final feature matrix."""
        df = df.copy()
        
        # Add all feature groups
        df = self._add_seed_features(df)
        df = self._add_away_neutral_features(df)
        df = self._add_shooting_features(df)
        df = self._add_conference_features(df)
        df = self._add_momentum_features(df)
        df = self._add_resume_features(df)
        df = self._add_composite_features(df)
        
        # Get available features
        available = [f for f in self.feature_cols if f in df.columns]
        missing = [f for f in self.feature_cols if f not in df.columns]
        
        if missing:
            print(f"  Missing features (filled with median): {missing}")
            for f in missing:
                df[f] = 0
        
        X = df[self.feature_cols].copy()
        
        # Handle missing values with median
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
    
    def _add_away_neutral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add away-neutral performance features."""
        if self.away_df is None:
            df['AWAY_NEUTRAL_EM'] = df.get('BADJ EM', df.get('KADJ EM', 0))
            df['HOME_AWAY_DROP'] = 0
            return df
        
        away_subset = self.away_df[['YEAR', 'TEAM', 'BADJ EM']].copy()
        away_subset.columns = ['YEAR', 'TEAM', 'AWAY_NEUTRAL_EM']
        
        df = df.merge(away_subset, on=['YEAR', 'TEAM'], how='left')
        
        # Calculate home-away drop (lower is better)
        home_em = df.get('BADJ EM', df.get('KADJ EM', 0))
        away_em = df['AWAY_NEUTRAL_EM']
        df['HOME_AWAY_DROP'] = home_em - away_em.fillna(home_em)
        df['AWAY_NEUTRAL_EM'] = df['AWAY_NEUTRAL_EM'].fillna(home_em)
        
        return df
    
    def _add_shooting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add shooting split features."""
        if self.shots_df is None:
            df['DUNKS_SHARE'] = 7  # avg
            df['CLOSE_TWOS_SHARE'] = 35  # avg
            return df
        
        shots_subset = self.shots_df[['YEAR', 'TEAM', 'DUNKS SHARE', 'CLOSE TWOS SHARE']].copy()
        shots_subset.columns = ['YEAR', 'TEAM', 'DUNKS_SHARE', 'CLOSE_TWOS_SHARE']
        
        df = df.merge(shots_subset, on=['YEAR', 'TEAM'], how='left')
        df['DUNKS_SHARE'] = df['DUNKS_SHARE'].fillna(df['DUNKS_SHARE'].median())
        df['CLOSE_TWOS_SHARE'] = df['CLOSE_TWOS_SHARE'].fillna(df['CLOSE_TWOS_SHARE'].median())
        
        return df
    
    def _add_conference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add conference historical success."""
        if self.conf_df is None:
            df['CONF_CHAMP_RATE'] = 0.5
            return df
        
        conf_subset = self.conf_df[['CONF', 'CHAMP_RATE']].copy()
        conf_subset.columns = ['CONF', 'CONF_CHAMP_RATE']
        
        df = df.merge(conf_subset, on='CONF', how='left')
        df['CONF_CHAMP_RATE'] = df['CONF_CHAMP_RATE'].fillna(0)
        
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
        """Add resume features (Q1 wins)."""
        if self.resume_df is None:
            df['Q1_WINS'] = 5
            return df
        
        resume_cols = ['YEAR', 'TEAM', 'Q1 W']
        resume_subset = self.resume_df[resume_cols].copy()
        resume_subset.columns = ['YEAR', 'TEAM', 'Q1_WINS']
        
        df = df.merge(resume_subset, on=['YEAR', 'TEAM'], how='left')
        df['Q1_WINS'] = df['Q1_WINS'].fillna(df['Q1_WINS'].median())
        
        return df
    
    def _add_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite features."""
        
        # EM x SOS
        if 'KADJ EM' in df.columns and 'ELITE SOS' in df.columns:
            sos_min = df['ELITE SOS'].min()
            sos_max = df['ELITE SOS'].max()
            sos_norm = (df['ELITE SOS'] - sos_min) / (sos_max - sos_min + 1e-8)
            df['EM_X_SOS'] = df['KADJ EM'] * (0.5 + sos_norm)
        else:
            df['EM_X_SOS'] = 0
        
        # Clutch composite
        if 'FT%' in df.columns and 'WAB' in df.columns:
            ft_norm = (df['FT%'] - df['FT%'].min()) / (df['FT%'].max() - df['FT%'].min() + 1e-8)
            wab_norm = (df['WAB'] - df['WAB'].min()) / (df['WAB'].max() - df['WAB'].min() + 1e-8)
            df['CLUTCH_COMPOSITE'] = 0.5 * ft_norm + 0.5 * wab_norm
        else:
            df['CLUTCH_COMPOSITE'] = 0
        
        # Rim Dominance: Dunks + Blocks (offensive + defensive rim presence)
        if 'DUNKS_SHARE' in df.columns and 'BLK%' in df.columns:
            dunks_norm = (df['DUNKS_SHARE'] - df['DUNKS_SHARE'].min()) / (df['DUNKS_SHARE'].max() - df['DUNKS_SHARE'].min() + 1e-8)
            blk_norm = (df['BLK%'] - df['BLK%'].min()) / (df['BLK%'].max() - df['BLK%'].min() + 1e-8)
            df['RIM_DOMINANCE'] = 0.5 * dunks_norm + 0.5 * blk_norm
        else:
            df['RIM_DOMINANCE'] = 0
            
        return df
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names if hasattr(self, 'feature_names') else []


def backtest_final():
    """Backtest the final model."""
    import sys
    sys.path.insert(0, '.')
    
    from src.data.loader import DataLoader
    from src.models.champion_model import ChampionPredictor
    from config.settings import FIRST_TEST_YEAR
    
    print("="*70)
    print("FINAL MODEL BACKTEST (All Features)")
    print("="*70)
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        builder = FinalFeatureBuilder()
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
    print("FINAL MODEL SUMMARY")
    print("="*70)
    print(f"Mean Champion Rank:   {np.mean(ranks):.2f}")
    print(f"Median Champion Rank: {np.median(ranks):.1f}")
    print(f"Top-1 Rate:  {sum(r==1 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-3 Rate:  {sum(r<=3 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-5 Rate:  {sum(r<=5 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-10 Rate: {sum(r<=10 for r in ranks)/len(ranks)*100:.1f}%")
    
    return results


if __name__ == "__main__":
    backtest_final()
