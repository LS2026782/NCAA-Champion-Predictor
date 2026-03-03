"""
Feature engineering for NCAA Championship Prediction.

This module handles:
- Selecting and validating feature columns
- Creating derived features (interactions, composites, era-relative stats)
- Handling missing values
- Feature scaling (for logistic regression)
- Concept-drift normalization (3-point revolution, transfer portal era)
- Cinderella detection (Low Talent / High Experience / High Efficiency)

INTERPRETABILITY NOTE:
Features are chosen to be meaningful to basketball analysts:
- Efficiency metrics explain team quality
- Four factors explain HOW teams win
- Experience/SOS explain tournament readiness
- Talent-Experience interaction captures the Blue Blood vs Cinderella dynamic
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    MODEL_FEATURES,
    EFFICIENCY_FEATURES,
    FOUR_FACTORS_OFF,
    FOUR_FACTORS_DEF,
    EXPERIENCE_FEATURES,
    SOS_FEATURES,
    SEED_CHAMPION_RATE,
)


class FeatureBuilder:
    """
    Builds feature matrices for champion prediction.
    
    Handles:
    - Feature selection from raw data
    - Derived feature creation
    - Missing value imputation
    - Optional scaling for logistic regression
    
    Attributes:
        scaler: StandardScaler fitted on training data
        feature_cols: List of feature column names
    """
    
    def __init__(self, feature_cols: Optional[List[str]] = None):
        """
        Initialize the feature builder.
        
        Args:
            feature_cols: List of features to use. If None, uses MODEL_FEATURES
        """
        self.feature_cols = feature_cols or MODEL_FEATURES
        self.scaler: Optional[StandardScaler] = None
        self._fitted = False
        # Stores all normalization parameters learned from the training set so
        # they can be reused without modification on the test set, preventing
        # data leakage from the test distribution into feature computation.
        self._fit_params: dict = {}
        
    def build_features(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Build feature matrix from raw data.
        
        Args:
            df: DataFrame with raw team statistics
            fit_scaler: If True, fit the scaler on this data (for training)
            
        Returns:
            Tuple of (DataFrame with features, numpy array of features)
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Add derived features — pass fit flag so normalization params are
        # learned from training data and reused on test data (no leakage).
        df = self._add_derived_features(df, fit=fit_scaler)
        
        # Select feature columns
        missing_features = [f for f in self.feature_cols if f not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features will be filled with 0: {missing_features}")
            for f in missing_features:
                df[f] = 0
                
        X = df[self.feature_cols].copy()
        
        # Handle missing values — store medians from training; reuse on test
        X = self._handle_missing(X, fit=fit_scaler)
        
        # Scale features
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self._fitted = True
        elif self.scaler is not None and self._fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        return df, X_scaled
    
    def _add_derived_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Add derived features to the DataFrame.

        When fit=True (training), normalization parameters (sos_min/max, fill
        medians for TALENT/EXP) are computed from the training distribution and
        stored in self._fit_params.  When fit=False (test/predict), the stored
        training-set parameters are reused, preventing test-set leakage.

        Features:
        - SEED_STRENGTH: Non-linear seed → historical champion rate (log-scaled)
        - EM_X_SOS: Efficiency margin × SOS interaction (training-normalized)
        - FOUR_FACTOR_OFF/DEF: Dean Oliver weighted composites
        - TALENT_X_EXP: Talent × Experience interaction
        - CINDERELLA: Binary flag for low-talent / high-exp / high-EM profile
        - EFG_MARGIN: EFG% offensive minus defensive
        - RELATIVE_3PR: 3-point rate relative to within-season average
        """
        # --- Non-linear seed mapping -------------------------------------------
        # Replace (17 - seed) with log of historical championship win-rate.
        # This correctly encodes the enormous jump from seed 1 → 2, and the
        # near-zero difference between seeds 8-16.
        if df['SEED'].dtype == 'object':
            df['SEED_NUM'] = df['SEED'].astype(str).str.extract(r'(\d+)').astype(float)
        else:
            df['SEED_NUM'] = df['SEED'].astype(float)

        def _seed_to_log_prob(seed_series: pd.Series) -> pd.Series:
            rate = seed_series.map(SEED_CHAMPION_RATE).fillna(SEED_CHAMPION_RATE[16])
            return np.log(rate)   # log-probability; negative, higher = better seed

        df['SEED_STRENGTH'] = _seed_to_log_prob(df['SEED_NUM'])

        # --- EM × SOS interaction (training-set normalization) -----------------
        if 'KADJ EM' in df.columns and 'ELITE SOS' in df.columns:
            if fit:
                self._fit_params['sos_min'] = df['ELITE SOS'].min()
                self._fit_params['sos_max'] = df['ELITE SOS'].max()
            sos_min = self._fit_params.get('sos_min', df['ELITE SOS'].min())
            sos_max = self._fit_params.get('sos_max', df['ELITE SOS'].max())
            sos_norm = (df['ELITE SOS'] - sos_min) / (sos_max - sos_min + 1e-8)
            df['EM_X_SOS'] = df['KADJ EM'] * (0.5 + sos_norm)
        else:
            df['EM_X_SOS'] = 0

        # --- Four factors composites (Dean Oliver weights) ----------------------
        if all(f in df.columns for f in ['EFG%', 'TOV%', 'OREB%', 'FTR']):
            df['FOUR_FACTOR_OFF'] = (
                0.40 * df['EFG%'] +
                0.25 * (100 - df['TOV%']) +
                0.20 * df['OREB%'] +
                0.15 * df['FTR']
            )

        if all(f in df.columns for f in ['EFG%D', 'TOV%D', 'DREB%', 'FTRD']):
            df['FOUR_FACTOR_DEF'] = (
                0.40 * (100 - df['EFG%D']) +
                0.25 * df['TOV%D'] +
                0.20 * df['DREB%'] +
                0.15 * (100 - df['FTRD'])
            )

        # --- Talent × Experience interaction (training-median fill) -------------
        if 'TALENT' in df.columns and 'EXP' in df.columns:
            if fit:
                t_med = df['TALENT'].median() if df['TALENT'].notna().any() else 30.0
                e_med = df['EXP'].median()    if df['EXP'].notna().any()    else 1.8
                self._fit_params['talent_median'] = t_med
                self._fit_params['exp_median']    = e_med
            talent_safe = df['TALENT'].fillna(self._fit_params.get('talent_median', 30.0))
            exp_safe    = df['EXP'].fillna(self._fit_params.get('exp_median',    1.8))
            df['TALENT_X_EXP'] = talent_safe * exp_safe
        else:
            df['TALENT_X_EXP'] = 0

        # --- Cinderella detector ------------------------------------------------
        if 'TALENT' in df.columns and 'EXP' in df.columns and 'KADJ EM' in df.columns:
            from src.models.champion_model import detect_cinderella_profile
            df['CINDERELLA'] = detect_cinderella_profile(
                talent=df['TALENT'].values,
                experience=df['EXP'].values,
                efficiency=df['KADJ EM'].values
            )
        else:
            df['CINDERELLA'] = 0

        # --- Shooting differential ----------------------------------------------
        if 'EFG%' in df.columns and 'EFG%D' in df.columns:
            df['EFG_MARGIN'] = df['EFG%'] - df['EFG%D']
        else:
            df['EFG_MARGIN'] = 0

        # --- Relative 3-point rate (within-season; no cross-season leakage) -----
        # Normalising within each season is the correct transform: we want to
        # know whether a team shoots more threes than its contemporaries, not
        # vs. the historical average (which would be contaminated by era drift).
        if '3PR' in df.columns and 'YEAR' in df.columns:
            season_avg = df.groupby('YEAR')['3PR'].transform('mean')
            season_std = df.groupby('YEAR')['3PR'].transform('std')
            df['RELATIVE_3PR'] = (df['3PR'] - season_avg) / (season_std + 1e-8)
        elif '3PT%' in df.columns and 'YEAR' in df.columns:
            season_avg = df.groupby('YEAR')['3PT%'].transform('mean')
            season_std = df.groupby('YEAR')['3PT%'].transform('std')
            df['RELATIVE_3PR'] = (df['3PT%'] - season_avg) / (season_std + 1e-8)
        else:
            df['RELATIVE_3PR'] = 0

        return df
    
    def _handle_missing(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Handle missing values in feature matrix.

        When fit=True: compute column medians from training data and store them
        in self._fit_params['imputation_medians'].
        When fit=False: reuse the stored training medians, preventing any
        information from the test distribution leaking into imputation.
        """
        if fit:
            self._fit_params['imputation_medians'] = {}

        medians = self._fit_params.get('imputation_medians', {})

        for col in X.columns:
            if X[col].isnull().any():
                if fit:
                    val = X[col].median()
                    if pd.isna(val):
                        val = 0.0
                    medians[col] = val
                else:
                    val = medians.get(col, 0.0)

                X[col] = X[col].fillna(val)

        if fit:
            self._fit_params['imputation_medians'] = medians

        return X
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_cols.copy()
    
    def get_feature_importance(
        self, 
        model, 
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract feature importance from a fitted model.
        
        Works with:
        - Logistic regression (uses coefficients)
        - Gradient boosting (uses feature_importances_)
        
        Args:
            model: Fitted sklearn model
            feature_names: Feature names (uses self.feature_cols if None)
            
        Returns:
            DataFrame with feature importance ranked
        """
        names = feature_names or self.feature_cols
        
        # Check for coefficients (logistic regression)
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            importance_df = pd.DataFrame({
                'feature': names,
                'coefficient': coefs,
                'abs_importance': np.abs(coefs)
            })
        # Check for feature_importances_ (tree-based)
        elif hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': names,
                'importance': model.feature_importances_,
                'abs_importance': model.feature_importances_
            })
        else:
            raise ValueError("Model doesn't have coef_ or feature_importances_")
            
        return importance_df.sort_values('abs_importance', ascending=False)
    
    def explain_prediction(
        self,
        model,
        team_features: np.ndarray,
        team_name: str,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Explain why a team is predicted as champion-like.
        
        Shows contribution of each feature to the prediction.
        
        Args:
            model: Fitted model
            team_features: Feature array for the team
            team_name: Name of team
            feature_names: Feature names
            
        Returns:
            DataFrame with feature contributions
        """
        names = feature_names or self.feature_cols
        
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            contributions = team_features * coefs
            
            explanation_df = pd.DataFrame({
                'feature': names,
                'value': team_features,
                'coefficient': coefs,
                'contribution': contributions
            })
            
            explanation_df['team'] = team_name
            return explanation_df.sort_values('contribution', ascending=False)
        else:
            # For tree-based models, can't do simple linear decomposition
            return pd.DataFrame({
                'feature': names,
                'value': team_features,
                'team': team_name
            })


def main():
    """Test feature builder functionality."""
    from src.data.loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_all()
    
    # Build features
    builder = FeatureBuilder()
    train_df, test_df = loader.get_train_test_split(2024)
    
    print("="*60)
    print("FEATURE BUILDING TEST")
    print("="*60)
    
    # Build training features (fit scaler)
    train_df_feat, X_train = builder.build_features(train_df, fit_scaler=True)
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Feature columns: {builder.get_feature_names()}")
    
    # Build test features (use fitted scaler)
    test_df_feat, X_test = builder.build_features(test_df, fit_scaler=False)
    print(f"\nTest features shape: {X_test.shape}")
    
    # Show derived features for champions
    print("\nDerived features for 2024 champion:")
    champ = test_df_feat[test_df_feat['IS_CHAMPION'] == 1]
    print(champ[['TEAM', 'SEED', 'SEED_STRENGTH', 'EM_X_SOS', 'KADJ EM', 'BARTHAG']].to_string())
    
    # Show feature summary
    print("\nFeature summary (training data):")
    feature_summary = pd.DataFrame({
        'feature': builder.get_feature_names(),
        'mean': X_train.mean(axis=0),
        'std': X_train.std(axis=0),
        'min': X_train.min(axis=0),
        'max': X_train.max(axis=0)
    })
    print(feature_summary.to_string())


if __name__ == "__main__":
    main()
