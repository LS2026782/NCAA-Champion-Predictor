"""
Enhanced Feature Engineering based on Champion Pattern Analysis.

New features discovered from deep analysis:
1. ELITE_COUNT - Number of elite (top 20%) categories
2. TOP5_OFFENSE - Binary: Is offense ranked top 5 in field?
3. TALENT_ELITE - Binary: Is talent rank top 15?
4. OREB_ELITE - Binary: Is OREB% top 10?
5. SEED_RISK - Penalty multiplier for lower seeds
6. SHOOTING_COMPOSITE - Combined eFG% and 3PT% score
7. BALANCE_SCORE - How balanced is offense vs defense?
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import MODEL_FEATURES


class EnhancedFeatureBuilder:
    """
    Enhanced feature builder with pattern-based features.
    """
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False
        
        # Base features from original model
        self.base_features = [
            'KADJ EM', 'KADJ O', 'KADJ D', 'BARTHAG',
            'EFG%', 'TOV%', 'OREB%', 'FTR',
            'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
            'ELITE SOS'
        ]
        
        # New pattern-based features
        self.derived_features = [
            'SEED_STRENGTH',
            'ELITE_COUNT',
            'TOP5_OFFENSE',
            'TOP5_EM',
            'TALENT_ELITE',
            'OREB_ELITE',
            'SEED_RISK',
            'SHOOTING_COMPOSITE',
            'BALANCE_SCORE',
            'EM_X_SOS',
            'WINS_ABOVE_BUBBLE'
        ]
        
    def build_features(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = False,
        all_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Build enhanced feature matrix.
        
        Args:
            df: DataFrame with team statistics
            fit_scaler: If True, fit the scaler
            all_data: Full dataset for computing relative ranks (optional)
            
        Returns:
            Tuple of (DataFrame with features, numpy array)
        """
        df = df.copy()
        
        # Use provided all_data or df for computing ranks
        rank_data = all_data if all_data is not None else df
        
        # Add derived features
        df = self._add_seed_features(df)
        df = self._add_elite_counts(df, rank_data)
        df = self._add_rank_features(df, rank_data)
        df = self._add_composite_features(df)
        
        # Combine all features
        all_features = self.base_features + self.derived_features
        available = [f for f in all_features if f in df.columns]
        
        X = df[available].copy()
        
        # Handle missing values with median
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Scale
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self._fitted = True
        elif self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        self.feature_names = available
        return df, X_scaled
    
    def _add_seed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seed-based features."""
        # Convert seed to numeric
        if df['SEED'].dtype == 'object':
            df['SEED_NUM'] = df['SEED'].astype(str).str.extract(r'(\d+)').astype(float)
        else:
            df['SEED_NUM'] = df['SEED'].astype(float)
        
        # Seed strength (higher = better)
        df['SEED_STRENGTH'] = 17 - df['SEED_NUM']
        
        # Seed risk penalty (exponential penalty for lower seeds)
        # Based on finding: 93.8% of champions are seeds 1-4
        df['SEED_RISK'] = np.where(
            df['SEED_NUM'] <= 4, 
            1.0,  # No penalty for top 4 seeds
            np.exp(-(df['SEED_NUM'] - 4) * 0.3)  # Exponential decay
        )
        
        return df
    
    def _add_elite_counts(self, df: pd.DataFrame, rank_data: pd.DataFrame) -> pd.DataFrame:
        """
        Count how many 'elite' categories each team is in.
        Champions average 3.6 elite categories.
        """
        elite_cols = ['KADJ EM', 'KADJ O', 'KADJ D', 'EFG%', 'EFG%D', 'OREB%', 'TALENT']
        lower_better = ['KADJ D', 'EFG%D']
        
        elite_counts = []
        
        for _, row in df.iterrows():
            year = row['YEAR']
            year_data = rank_data[rank_data['YEAR'] == year]
            
            count = 0
            for col in elite_cols:
                if col not in year_data.columns:
                    continue
                    
                if col in lower_better:
                    threshold = year_data[col].quantile(0.20)
                    if row[col] <= threshold:
                        count += 1
                else:
                    threshold = year_data[col].quantile(0.80)
                    if row[col] >= threshold:
                        count += 1
                        
            elite_counts.append(count)
            
        df['ELITE_COUNT'] = elite_counts
        return df
    
    def _add_rank_features(self, df: pd.DataFrame, rank_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add within-year rank features.
        Based on finding: 69% of champions have Top-5 offense.
        """
        # Compute ranks within year
        features_to_rank = {
            'KADJ O': False,   # Higher is better
            'KADJ EM': False,
            'KADJ D': True,    # Lower is better
            'OREB%': False,
            'TALENT': False
        }
        
        for col, ascending in features_to_rank.items():
            if col not in df.columns:
                continue
                
            ranks = []
            for _, row in df.iterrows():
                year_data = rank_data[rank_data['YEAR'] == row['YEAR']]
                if ascending:
                    rank = (year_data[col] < row[col]).sum() + 1
                else:
                    rank = (year_data[col] > row[col]).sum() + 1
                ranks.append(rank)
            
            df[f'{col}_RANK'] = ranks
        
        # Binary elite features
        if 'KADJ O_RANK' in df.columns:
            df['TOP5_OFFENSE'] = (df['KADJ O_RANK'] <= 5).astype(int)
            
        if 'KADJ EM_RANK' in df.columns:
            df['TOP5_EM'] = (df['KADJ EM_RANK'] <= 5).astype(int)
            
        if 'TALENT_RANK' in df.columns:
            df['TALENT_ELITE'] = (df['TALENT_RANK'] <= 15).astype(int)
            
        if 'OREB%_RANK' in df.columns:
            df['OREB_ELITE'] = (df['OREB%_RANK'] <= 10).astype(int)
            
        return df
    
    def _add_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite/interaction features."""
        
        # Shooting composite (eFG% and 3PT% both matter)
        if 'EFG%' in df.columns and '3PT%' in df.columns:
            df['SHOOTING_COMPOSITE'] = (
                0.6 * df['EFG%'] + 0.4 * df['3PT%']
            )
        
        # Balance score: how close is offense rank to defense rank?
        # Champions tend to be balanced or offense-dominant
        if 'KADJ O_RANK' in df.columns and 'KADJ D_RANK' in df.columns:
            # Lower is better (more balanced)
            df['BALANCE_SCORE'] = np.abs(df['KADJ O_RANK'] - df['KADJ D_RANK'])
            # Penalize defense-dominant teams (offense rank > defense rank)
            df['BALANCE_SCORE'] = np.where(
                df['KADJ O_RANK'] > df['KADJ D_RANK'],
                df['BALANCE_SCORE'] * 1.5,  # Penalty for weak offense
                df['BALANCE_SCORE']
            )
        
        # EM x SOS interaction
        if 'KADJ EM' in df.columns and 'ELITE SOS' in df.columns:
            sos_min = df['ELITE SOS'].min()
            sos_max = df['ELITE SOS'].max()
            sos_norm = (df['ELITE SOS'] - sos_min) / (sos_max - sos_min + 1e-8)
            df['EM_X_SOS'] = df['KADJ EM'] * (0.5 + sos_norm)
        
        # Wins Above Bubble (WAB) if available
        if 'WAB' in df.columns:
            df['WINS_ABOVE_BUBBLE'] = df['WAB']
        else:
            df['WINS_ABOVE_BUBBLE'] = 0
            
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names if hasattr(self, 'feature_names') else []


def main():
    """Test enhanced feature builder."""
    import sys
    sys.path.insert(0, '.')
    
    from src.data.loader import DataLoader
    from src.models.champion_model import ChampionPredictor
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    # Test on 2024
    train_df, test_df = loader.get_train_test_split(2024)
    
    # Build enhanced features
    print("\nBuilding enhanced features...")
    builder = EnhancedFeatureBuilder()
    train_df_feat, X_train = builder.build_features(train_df, fit_scaler=True, all_data=full_data)
    test_df_feat, X_test = builder.build_features(test_df, fit_scaler=False, all_data=full_data)
    
    print(f"Features: {builder.get_feature_names()}")
    print(f"Feature count: {len(builder.get_feature_names())}")
    
    y_train = train_df_feat['IS_CHAMPION'].values
    
    # Train model
    print("\nTraining enhanced model...")
    model = ChampionPredictor(model_type='logreg')
    model.fit(X_train, y_train, feature_names=builder.get_feature_names())
    
    # Predict
    probs = model.predict_proba(X_test)
    test_df_feat = test_df_feat.copy()
    test_df_feat['PROB'] = probs
    test_df_feat['RANK'] = test_df_feat['PROB'].rank(ascending=False).astype(int)
    
    # Results
    print("\n" + "="*60)
    print("ENHANCED MODEL - 2024 PREDICTIONS")
    print("="*60)
    
    print("\nTop 10:")
    for _, row in test_df_feat.nlargest(10, 'PROB').iterrows():
        champ = " ** CHAMPION **" if row['IS_CHAMPION'] == 1 else ""
        print(f"  {int(row['RANK']):2d}. {row['TEAM']:<20} (Seed {int(row['SEED'])}) "
              f"Prob={row['PROB']:.3f} Elite={int(row['ELITE_COUNT'])}{champ}")
    
    # Champion rank
    champ_row = test_df_feat[test_df_feat['IS_CHAMPION'] == 1].iloc[0]
    print(f"\nActual champion '{champ_row['TEAM']}' ranked #{int(champ_row['RANK'])}")
    
    # Feature importance
    print("\nTop feature coefficients:")
    coefs = model.get_coefficients()
    print(coefs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
