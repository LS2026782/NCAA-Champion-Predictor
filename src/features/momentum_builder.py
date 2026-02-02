"""
Momentum-Based Feature Engineering

New features:
1. MOMENTUM_EM - Change in efficiency from preseason to final
2. MOMENTUM_RANK - Rank improvement from preseason
3. CONF_TOURNEY_CHAMP - Did team win conference tournament (Auto bid)?
4. Q1_WINS - Quality wins against top opponents
5. LATE_SEASON_STRENGTH - Composite of momentum indicators
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import RAW_DATA_DIR


class MomentumFeatureBuilder:
    """
    Feature builder with momentum and conference tournament features.
    """
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False
        
        # Load auxiliary data
        self._load_auxiliary_data()
        
        # Core efficiency features
        self.base_features = [
            'KADJ EM', 'KADJ O', 'KADJ D', 'BARTHAG',
            'EFG%', 'TOV%', 'OREB%', 'FTR',
            'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
            'ELITE SOS', 'TALENT'
        ]
        
        # New momentum features
        self.momentum_features = [
            'SEED_STRENGTH',
            'MOMENTUM_EM',           # EM improvement from preseason
            'MOMENTUM_RANK',         # Rank improvement from preseason
            'CONF_TOURNEY_CHAMP',    # Won conference tournament
            'Q1_WINS',               # Quality wins
            'Q1_Q2_WINS',            # Quality wins (Q1 + Q2)
            'LATE_SEASON_COMPOSITE', # Combined momentum score
            'EM_X_SOS',
        ]
        
    def _load_auxiliary_data(self):
        """Load preseason and resume data for momentum features."""
        try:
            self.preseason_df = pd.read_csv(RAW_DATA_DIR / 'KenPom Preseason.csv')
            print(f"  Loaded preseason data: {len(self.preseason_df)} records")
        except Exception as e:
            print(f"  Warning: Could not load preseason data: {e}")
            self.preseason_df = None
            
        try:
            self.resume_df = pd.read_csv(RAW_DATA_DIR / 'Resumes.csv')
            print(f"  Loaded resume data: {len(self.resume_df)} records")
        except Exception as e:
            print(f"  Warning: Could not load resume data: {e}")
            self.resume_df = None
    
    def build_features(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = False,
        all_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Build feature matrix with momentum features.
        """
        df = df.copy()
        
        # Add seed features
        df = self._add_seed_features(df)
        
        # Add momentum features from preseason data
        df = self._add_momentum_features(df)
        
        # Add conference tournament features from resume data
        df = self._add_conf_tourney_features(df)
        
        # Add composite momentum score
        df = self._add_composite_features(df)
        
        # Combine all features
        all_features = self.base_features + self.momentum_features
        available = [f for f in all_features if f in df.columns]
        
        # Fill missing with defaults
        for f in all_features:
            if f not in df.columns:
                df[f] = 0
                
        X = df[all_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median() if X[col].median() == X[col].median() else 0)
        
        # Scale
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self._fitted = True
        elif self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        self.feature_names = all_features
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
        """
        Add momentum features from preseason data.
        
        MOMENTUM_EM: How much did team improve from preseason?
        MOMENTUM_RANK: How many spots did team climb in rankings?
        """
        if self.preseason_df is None:
            df['MOMENTUM_EM'] = 0
            df['MOMENTUM_RANK'] = 0
            return df
        
        # Merge preseason data
        preseason_cols = ['YEAR', 'TEAM', 'KADJ EM CHANGE', 'KADJ EM RANK CHANGE']
        preseason_subset = self.preseason_df[preseason_cols].copy()
        preseason_subset.columns = ['YEAR', 'TEAM', 'MOMENTUM_EM', 'MOMENTUM_RANK']
        
        df = df.merge(preseason_subset, on=['YEAR', 'TEAM'], how='left')
        
        # Fill missing with median (teams without preseason data)
        df['MOMENTUM_EM'] = df['MOMENTUM_EM'].fillna(df['MOMENTUM_EM'].median())
        df['MOMENTUM_RANK'] = df['MOMENTUM_RANK'].fillna(0)
        
        return df
    
    def _add_conf_tourney_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add conference tournament features from resume data.
        
        CONF_TOURNEY_CHAMP: 1 if team won conference tournament (Auto bid)
        Q1_WINS: Number of Quadrant 1 wins
        Q1_Q2_WINS: Number of Q1 + Q2 wins
        """
        if self.resume_df is None:
            df['CONF_TOURNEY_CHAMP'] = 0
            df['Q1_WINS'] = 0
            df['Q1_Q2_WINS'] = 0
            return df
        
        # Merge resume data
        resume_cols = ['YEAR', 'TEAM', 'BID TYPE', 'Q1 W', 'Q1 PLUS Q2 W']
        resume_subset = self.resume_df[resume_cols].copy()
        
        # Create binary conference tournament champion feature
        resume_subset['CONF_TOURNEY_CHAMP'] = (resume_subset['BID TYPE'] == 'Auto').astype(int)
        resume_subset['Q1_WINS'] = resume_subset['Q1 W']
        resume_subset['Q1_Q2_WINS'] = resume_subset['Q1 PLUS Q2 W']
        
        resume_subset = resume_subset[['YEAR', 'TEAM', 'CONF_TOURNEY_CHAMP', 'Q1_WINS', 'Q1_Q2_WINS']]
        
        df = df.merge(resume_subset, on=['YEAR', 'TEAM'], how='left')
        
        # Fill missing
        df['CONF_TOURNEY_CHAMP'] = df['CONF_TOURNEY_CHAMP'].fillna(0)
        df['Q1_WINS'] = df['Q1_WINS'].fillna(df['Q1_WINS'].median())
        df['Q1_Q2_WINS'] = df['Q1_Q2_WINS'].fillna(df['Q1_Q2_WINS'].median())
        
        return df
    
    def _add_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite features combining multiple signals.
        """
        # Late season composite: combines momentum + quality wins + conf tourney
        # Normalize each component to 0-1 range then combine
        
        if 'MOMENTUM_EM' in df.columns:
            mom_min, mom_max = df['MOMENTUM_EM'].min(), df['MOMENTUM_EM'].max()
            mom_norm = (df['MOMENTUM_EM'] - mom_min) / (mom_max - mom_min + 1e-8)
        else:
            mom_norm = 0
            
        if 'Q1_Q2_WINS' in df.columns:
            q_min, q_max = df['Q1_Q2_WINS'].min(), df['Q1_Q2_WINS'].max()
            q_norm = (df['Q1_Q2_WINS'] - q_min) / (q_max - q_min + 1e-8)
        else:
            q_norm = 0
            
        conf_champ = df.get('CONF_TOURNEY_CHAMP', 0)
        
        # Weighted composite: momentum (40%) + quality wins (40%) + conf tourney (20%)
        df['LATE_SEASON_COMPOSITE'] = (
            0.40 * mom_norm + 
            0.40 * q_norm + 
            0.20 * conf_champ
        )
        
        # EM x SOS interaction
        if 'KADJ EM' in df.columns and 'ELITE SOS' in df.columns:
            sos_min = df['ELITE SOS'].min()
            sos_max = df['ELITE SOS'].max()
            sos_norm = (df['ELITE SOS'] - sos_min) / (sos_max - sos_min + 1e-8)
            df['EM_X_SOS'] = df['KADJ EM'] * (0.5 + sos_norm)
        else:
            df['EM_X_SOS'] = 0
            
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names if hasattr(self, 'feature_names') else []


def analyze_momentum_patterns():
    """Analyze how momentum features relate to champions."""
    print("="*70)
    print("MOMENTUM PATTERN ANALYSIS")
    print("="*70)
    
    # Load data
    kp = pd.read_csv(RAW_DATA_DIR / 'KenPom Barttorvik.csv')
    kp = kp[(kp['YEAR'] >= 2008) & (kp['YEAR'] <= 2024) & (kp['SEED'].notna())]
    kp['IS_CHAMPION'] = (kp['ROUND'] == 1).astype(int)
    
    preseason = pd.read_csv(RAW_DATA_DIR / 'KenPom Preseason.csv')
    resume = pd.read_csv(RAW_DATA_DIR / 'Resumes.csv')
    
    # Merge
    df = kp.merge(
        preseason[['YEAR', 'TEAM', 'KADJ EM CHANGE', 'KADJ EM RANK CHANGE']],
        on=['YEAR', 'TEAM'],
        how='left'
    )
    df = df.merge(
        resume[['YEAR', 'TEAM', 'BID TYPE', 'Q1 W', 'Q1 PLUS Q2 W']],
        on=['YEAR', 'TEAM'],
        how='left'
    )
    
    df['CONF_TOURNEY_CHAMP'] = (df['BID TYPE'] == 'Auto').astype(int)
    
    champs = df[df['IS_CHAMPION'] == 1]
    non_champs = df[df['IS_CHAMPION'] == 0]
    
    print("\n--- MOMENTUM ANALYSIS ---")
    print(f"{'Metric':<25} {'Champions':>12} {'Field':>12} {'Diff':>10}")
    print("-"*60)
    
    metrics = [
        ('KADJ EM CHANGE', False),
        ('KADJ EM RANK CHANGE', False),
        ('CONF_TOURNEY_CHAMP', False),
        ('Q1 W', False),
        ('Q1 PLUS Q2 W', False)
    ]
    
    for metric, lower_better in metrics:
        if metric in df.columns:
            c_mean = champs[metric].mean()
            f_mean = non_champs[metric].mean()
            diff = c_mean - f_mean
            print(f"{metric:<25} {c_mean:>12.2f} {f_mean:>12.2f} {diff:>+10.2f}")
    
    print("\n--- CONFERENCE TOURNAMENT CHAMPION RATE ---")
    champ_auto = (champs['BID TYPE'] == 'Auto').sum()
    total_champs = len(champs[champs['BID TYPE'].notna()])
    print(f"Champions who won conf tourney: {champ_auto}/{total_champs} ({champ_auto/total_champs*100:.1f}%)")
    
    field_auto = (non_champs['BID TYPE'] == 'Auto').sum()
    total_field = len(non_champs[non_champs['BID TYPE'].notna()])
    print(f"Field who won conf tourney: {field_auto}/{total_field} ({field_auto/total_field*100:.1f}%)")


def main():
    """Test momentum feature builder."""
    import sys
    sys.path.insert(0, '.')
    
    from src.data.loader import DataLoader
    from src.models.champion_model import ChampionPredictor
    
    # Analyze patterns first
    analyze_momentum_patterns()
    
    # Load data
    print("\n" + "="*70)
    print("MOMENTUM MODEL TEST")
    print("="*70)
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    train_df, test_df = loader.get_train_test_split(2024)
    
    # Build features
    print("\nBuilding momentum features...")
    builder = MomentumFeatureBuilder()
    train_df_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
    test_df_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
    
    print(f"Features ({len(builder.get_feature_names())}): {builder.get_feature_names()}")
    
    y_train = train_df_feat['IS_CHAMPION'].values
    
    # Train model
    print("\nTraining momentum model...")
    model = ChampionPredictor(model_type='logreg')
    model.fit(X_train, y_train, feature_names=builder.get_feature_names())
    
    # Predict
    probs = model.predict_proba(X_test)
    test_df_feat = test_df_feat.copy()
    test_df_feat['PROB'] = probs
    test_df_feat['RANK'] = test_df_feat['PROB'].rank(ascending=False).astype(int)
    
    # Results
    print("\n" + "="*60)
    print("MOMENTUM MODEL - 2024 PREDICTIONS")
    print("="*60)
    
    print("\nTop 10:")
    for _, row in test_df_feat.nlargest(10, 'PROB').iterrows():
        champ = " ** CHAMPION **" if row['IS_CHAMPION'] == 1 else ""
        mom = row.get('MOMENTUM_EM', 0)
        conf = "Auto" if row.get('CONF_TOURNEY_CHAMP', 0) == 1 else "At-Large"
        print(f"  {int(row['RANK']):2d}. {row['TEAM']:<20} Seed={int(row['SEED'])} "
              f"Mom={mom:+.1f} Conf={conf}{champ}")
    
    # Champion rank
    champ_row = test_df_feat[test_df_feat['IS_CHAMPION'] == 1].iloc[0]
    print(f"\nActual champion '{champ_row['TEAM']}' ranked #{int(champ_row['RANK'])}")
    
    # Feature importance
    print("\nTop feature coefficients:")
    coefs = model.get_coefficients()
    print(coefs.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
