"""
Champion prediction models.

This module implements:
- Logistic Regression with L2 regularization (interpretable baseline)
- Gradient Boosting with strict regularization (optional ensemble)
- Probability calibration for well-calibrated outputs

DESIGN PHILOSOPHY:
- Interpretability over marginal accuracy gains
- Strong regularization to prevent overfitting on small champion sample
- Class balancing to handle extreme imbalance (~1:67 ratio)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import LOGREG_CONFIG, GBM_CONFIG


class ChampionPredictor:
    """
    Predicts championship probability for NCAA tournament teams.
    
    Supports two model types:
    1. Logistic Regression (default) - interpretable, well-calibrated
    2. Gradient Boosting - potentially higher accuracy, less interpretable
    
    Attributes:
        model_type: 'logreg' or 'gbm'
        model: Fitted sklearn model
        calibrated_model: Optional calibrated version
        feature_names: Names of input features
    """
    
    def __init__(
        self, 
        model_type: str = 'logreg',
        calibrate: bool = False
    ):
        """
        Initialize the champion predictor.
        
        Args:
            model_type: 'logreg' for logistic regression, 'gbm' for gradient boosting
            calibrate: Whether to apply probability calibration
        """
        self.model_type = model_type
        self.calibrate = calibrate
        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        self._fitted = False
        
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[list] = None
    ) -> 'ChampionPredictor':
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (1 = champion, 0 = not champion)
            feature_names: Names of features for interpretability
            
        Returns:
            self
        """
        self.feature_names = feature_names
        
        if self.model_type == 'logreg':
            self._fit_logistic_regression(X, y)
        elif self.model_type == 'gbm':
            self._fit_gradient_boosting(X, y)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        if self.calibrate:
            self._calibrate_probabilities(X, y)
            
        self._fitted = True
        return self
    
    def _fit_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit logistic regression with cross-validated regularization.
        
        Uses L2 regularization with automatic strength selection via CV.
        Class weights are balanced to handle imbalance.
        """
        print(f"Fitting Logistic Regression (n={len(y)}, positives={y.sum()})")
        
        # Use LogisticRegressionCV for automatic regularization tuning
        self.model = LogisticRegressionCV(
            Cs=np.logspace(-4, 2, LOGREG_CONFIG['Cs']),
            cv=min(LOGREG_CONFIG['cv'], int(y.sum())),  # CV folds <= positives
            penalty=LOGREG_CONFIG['penalty'],
            class_weight=LOGREG_CONFIG['class_weight'],
            scoring=LOGREG_CONFIG['scoring'],
            max_iter=LOGREG_CONFIG['max_iter'],
            random_state=LOGREG_CONFIG['random_state'],
            solver='lbfgs'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
            
        print(f"  Best C (regularization): {self.model.C_[0]:.4f}")
        
    def _fit_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit gradient boosting with strict regularization.
        
        Uses shallow trees and strong regularization to prevent overfitting.
        """
        print(f"Fitting Gradient Boosting (n={len(y)}, positives={y.sum()})")
        
        # Calculate class weight for balancing
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        # Use HistGradientBoostingClassifier for efficiency
        # Note: doesn't support class_weight directly, use sample_weight
        self.model = HistGradientBoostingClassifier(
            max_iter=GBM_CONFIG['max_iter'],
            max_depth=GBM_CONFIG['max_depth'],
            min_samples_leaf=GBM_CONFIG['min_samples_leaf'],
            l2_regularization=GBM_CONFIG['l2_regularization'],
            learning_rate=GBM_CONFIG['learning_rate'],
            early_stopping=GBM_CONFIG['early_stopping'],
            validation_fraction=GBM_CONFIG['validation_fraction'],
            n_iter_no_change=GBM_CONFIG['n_iter_no_change'],
            random_state=GBM_CONFIG['random_state']
        )
        
        # Create sample weights for class balancing
        sample_weight = np.where(y == 1, scale_pos_weight, 1.0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y, sample_weight=sample_weight)
            
        print(f"  Iterations used: {self.model.n_iter_}")
        
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Apply probability calibration using isotonic regression.
        
        Useful when probabilities need to be well-calibrated for
        comparison across years or simulation.
        """
        print("Applying probability calibration...")
        
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.model,
            method='isotonic',
            cv=min(3, int(y.sum()))  # Use fewer folds due to small positive class
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.calibrated_model.fit(X, y)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict championship probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities (probability of being champion)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if self.calibrated_model is not None:
            probs = self.calibrated_model.predict_proba(X)
        else:
            probs = self.model.predict_proba(X)
            
        # Return probability of positive class (champion)
        return probs[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary champion labels.
        
        Args:
            X: Feature matrix
            threshold: Probability threshold for positive prediction
            
        Returns:
            Binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def get_coefficients(self) -> Optional[pd.DataFrame]:
        """
        Get model coefficients (logistic regression only).
        
        Returns:
            DataFrame with feature names and coefficients, or None for GBM
        """
        if self.model_type != 'logreg':
            return None
            
        if self.feature_names is None:
            names = [f'feature_{i}' for i in range(len(self.model.coef_[0]))]
        else:
            names = self.feature_names
            
        coefs = self.model.coef_[0]
        
        df = pd.DataFrame({
            'feature': names,
            'coefficient': coefs,
            'odds_ratio': np.exp(coefs),
            'abs_coef': np.abs(coefs)
        })
        
        return df.sort_values('abs_coef', ascending=False)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance for any model type.
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_names is None:
            n_features = (self.model.coef_.shape[1] if hasattr(self.model, 'coef_') 
                         else len(self.model.feature_importances_))
            names = [f'feature_{i}' for i in range(n_features)]
        else:
            names = self.feature_names
            
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            raise ValueError("Model doesn't have interpretable importance")
            
        df = pd.DataFrame({
            'feature': names,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False)
    
    def explain_team(
        self, 
        X: np.ndarray, 
        team_name: str
    ) -> pd.DataFrame:
        """
        Explain prediction for a specific team.
        
        Shows how each feature contributes to the championship probability.
        Only works for logistic regression.
        
        Args:
            X: Feature values for the team (1D array)
            team_name: Name of the team
            
        Returns:
            DataFrame with feature contributions
        """
        if self.model_type != 'logreg':
            print("Warning: Detailed explanation only available for logistic regression")
            return pd.DataFrame()
            
        if self.feature_names is None:
            names = [f'feature_{i}' for i in range(len(X))]
        else:
            names = self.feature_names
            
        coefs = self.model.coef_[0]
        contributions = X * coefs
        
        # Get probability
        prob = self.predict_proba(X.reshape(1, -1))[0]
        
        df = pd.DataFrame({
            'feature': names,
            'value': X,
            'coefficient': coefs,
            'contribution': contributions,
            'direction': ['+' if c > 0 else '-' for c in contributions]
        })
        
        df = df.sort_values('contribution', ascending=False)
        df['team'] = team_name
        df['probability'] = prob
        
        return df


def main():
    """Test model training functionality."""
    from src.data.loader import DataLoader
    from src.features.builder import FeatureBuilder
    
    # Load data
    loader = DataLoader()
    df = loader.load_all()
    
    # Get train/test split
    train_df, test_df = loader.get_train_test_split(2024)
    
    # Build features
    builder = FeatureBuilder()
    train_df_feat, X_train = builder.build_features(train_df, fit_scaler=True)
    test_df_feat, X_test = builder.build_features(test_df, fit_scaler=False)
    
    y_train = train_df_feat['IS_CHAMPION'].values
    y_test = test_df_feat['IS_CHAMPION'].values
    
    print("="*60)
    print("MODEL TRAINING TEST")
    print("="*60)
    
    # Train logistic regression
    print("\n--- Logistic Regression ---")
    logreg = ChampionPredictor(model_type='logreg')
    logreg.fit(X_train, y_train, feature_names=builder.get_feature_names())
    
    # Get coefficients
    print("\nTop feature coefficients:")
    coefs = logreg.get_coefficients()
    print(coefs.head(10).to_string())
    
    # Predict on test set
    test_probs = logreg.predict_proba(X_test)
    test_df_feat['PROB'] = test_probs
    
    print("\nTop 10 predicted champions for 2024:")
    top_10 = test_df_feat.nlargest(10, 'PROB')[['TEAM', 'SEED', 'PROB', 'IS_CHAMPION']]
    print(top_10.to_string())
    
    # Find where actual champion ranks
    test_df_feat['RANK'] = test_df_feat['PROB'].rank(ascending=False)
    champ = test_df_feat[test_df_feat['IS_CHAMPION'] == 1]
    print(f"\nActual champion rank: {int(champ['RANK'].values[0])}")
    print(f"Champion probability: {champ['PROB'].values[0]:.4f}")
    
    # Explain champion prediction
    print("\nExplanation for actual champion:")
    champ_idx = test_df_feat[test_df_feat['IS_CHAMPION'] == 1].index[0]
    champ_X = X_test[test_df_feat.index.get_loc(champ_idx)]
    explanation = logreg.explain_team(champ_X, "Connecticut")
    print(explanation[['feature', 'value', 'coefficient', 'contribution', 'direction']].head(8).to_string())
    
    # Train gradient boosting
    print("\n--- Gradient Boosting ---")
    gbm = ChampionPredictor(model_type='gbm')
    gbm.fit(X_train, y_train, feature_names=builder.get_feature_names())
    
    # Feature importance
    print("\nTop feature importance:")
    importance = gbm.get_feature_importance()
    print(importance.head(10).to_string())
    
    # Predict
    gbm_probs = gbm.predict_proba(X_test)
    test_df_feat['GBM_PROB'] = gbm_probs
    test_df_feat['GBM_RANK'] = test_df_feat['GBM_PROB'].rank(ascending=False)
    
    print("\nTop 10 by GBM:")
    top_10_gbm = test_df_feat.nlargest(10, 'GBM_PROB')[['TEAM', 'SEED', 'GBM_PROB', 'IS_CHAMPION']]
    print(top_10_gbm.to_string())
    
    champ_gbm = test_df_feat[test_df_feat['IS_CHAMPION'] == 1]
    print(f"\nGBM champion rank: {int(champ_gbm['GBM_RANK'].values[0])}")


if __name__ == "__main__":
    main()
