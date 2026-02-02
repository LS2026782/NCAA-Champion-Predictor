"""
Feature engineering for NCAA Championship Prediction.

This module handles:
- Selecting and validating feature columns
- Creating derived features (interactions, composites)
- Handling missing values
- Feature scaling (for logistic regression)

INTERPRETABILITY NOTE:
Features are chosen to be meaningful to basketball analysts:
- Efficiency metrics explain team quality
- Four factors explain HOW teams win
- Experience/SOS explain tournament readiness
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
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Select feature columns
        available_features = [f for f in self.feature_cols if f in df.columns]
        missing_features = [f for f in self.feature_cols if f not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features will be filled with 0: {missing_features}")
            for f in missing_features:
                df[f] = 0
                
        X = df[self.feature_cols].copy()
        
        # Handle missing values
        X = self._handle_missing(X)
        
        # Scale features if requested
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self._fitted = True
        elif self.scaler is not None and self._fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        return df, X_scaled
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to the DataFrame.
        
        Derived features:
        - SEED_STRENGTH: Inverted seed (17 - seed) so higher = better
        - EM_X_SOS: Efficiency margin * SOS interaction
        - FOUR_FACTOR_OFF_COMPOSITE: Weighted offensive four factors
        - FOUR_FACTOR_DEF_COMPOSITE: Weighted defensive four factors
        """
        # Seed strength (higher is better seed)
        # Convert seed string to numeric if needed
        if df['SEED'].dtype == 'object':
            # Handle seeds like "11a" or "11b" (First Four)
            df['SEED_NUM'] = df['SEED'].astype(str).str.extract(r'(\d+)').astype(float)
        else:
            df['SEED_NUM'] = df['SEED'].astype(float)
            
        df['SEED_STRENGTH'] = 17 - df['SEED_NUM']
        
        # Efficiency margin * SOS interaction
        # Teams with high EM against strong schedules are more impressive
        if 'KADJ EM' in df.columns and 'ELITE SOS' in df.columns:
            # Normalize SOS to 0-1 range for the interaction
            sos_min = df['ELITE SOS'].min()
            sos_max = df['ELITE SOS'].max()
            sos_normalized = (df['ELITE SOS'] - sos_min) / (sos_max - sos_min + 1e-8)
            df['EM_X_SOS'] = df['KADJ EM'] * (0.5 + sos_normalized)
        else:
            df['EM_X_SOS'] = 0
            
        # Four factors composite (offensive)
        # Weights based on Dean Oliver's research: eFG% most important
        if all(f in df.columns for f in ['EFG%', 'TOV%', 'OREB%', 'FTR']):
            df['FOUR_FACTOR_OFF'] = (
                0.40 * df['EFG%'] +
                0.25 * (100 - df['TOV%']) +  # Invert: lower TO% is better
                0.20 * df['OREB%'] +
                0.15 * df['FTR']
            )
        
        # Four factors composite (defensive)
        if all(f in df.columns for f in ['EFG%D', 'TOV%D', 'DREB%', 'FTRD']):
            df['FOUR_FACTOR_DEF'] = (
                0.40 * (100 - df['EFG%D']) +  # Invert: lower opponent eFG% is better
                0.25 * df['TOV%D'] +           # Higher opponent TO% is better
                0.20 * df['DREB%'] +
                0.15 * (100 - df['FTRD'])     # Invert: lower opponent FTR is better
            )
            
        return df
    
    def _handle_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in feature matrix.
        
        Strategy: Fill with column median (robust to outliers)
        """
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"  Filled {X[col].isnull().sum()} missing values in {col} with median {median_val:.2f}")
                
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
