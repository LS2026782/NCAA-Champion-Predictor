"""
Champion prediction models.

This module implements:
- Logistic Regression with L2 regularization (interpretable baseline)
- Gradient Boosting with strict regularization (non-linear interactions)
- Probability calibration for well-calibrated outputs
- Temporal cross-validation (GroupKFold by season) to prevent leakage
- Cinderella detection via the Talent-Experience interaction
- Concept-drift-aware training with optional era weighting

DESIGN PHILOSOPHY:
- Interpretability over marginal accuracy gains
- Strong regularization to prevent overfitting on small champion sample (~22 positives)
- Class balancing to handle extreme imbalance (~1:67 ratio)
- Temporal integrity: training data NEVER leaks future season statistics
- Longitudinal learning: the model sees the full 2002-2025 evolution of the game
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, GroupKFold
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import LOGREG_CONFIG, GBM_CONFIG, OPTUNA_CONFIG


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
        self._season_groups = None
        self._era_weights = None
        
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[list] = None,
        season_groups: Optional[np.ndarray] = None,
        era_weights: Optional[np.ndarray] = None
    ) -> 'ChampionPredictor':
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (1 = champion, 0 = not champion)
            feature_names: Names of features for interpretability
            season_groups: Season year for each row, used for GroupKFold CV
                          to prevent temporal leakage within cross-validation
            era_weights: Optional per-sample weights for concept drift.
                        More recent seasons can be up-weighted to prioritize
                        the modern game's characteristics.
            
        Returns:
            self
        """
        self.feature_names = feature_names
        self._season_groups = season_groups
        self._era_weights = era_weights
        
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
        Class weights are balanced to handle the ~1:67 champion imbalance.
        
        When season groups are provided, uses GroupKFold to prevent temporal
        leakage (teams from the same season never appear in both train and
        validation within the CV loop).
        """
        print(f"Fitting Logistic Regression (n={len(y)}, positives={y.sum()})")
        
        n_folds = min(LOGREG_CONFIG['cv'], int(y.sum()))
        
        cv_strategy = n_folds
        if self._season_groups is not None:
            unique_seasons = np.unique(self._season_groups)
            n_groups = len(unique_seasons)
            if n_groups >= 3:
                cv_strategy = GroupKFold(n_splits=min(n_folds, n_groups))
                print(f"  Using GroupKFold with {min(n_folds, n_groups)} season groups")

        # Pre-compute GroupKFold splits into a list of (train, test) tuples so we
        # don't need to pass groups= to fit() (requires metadata routing in newer sklearn).
        # IMPORTANT: use pre-computed splits only when NOT calibrating — CalibratedClassifierCV
        # creates subsets of X and will call fit() on those subsets, where the original
        # index-based splits would be out of bounds.  When calibrating, fall back to an
        # integer CV count so LogisticRegressionCV creates fresh splits for each subset.
        if self._season_groups is not None and isinstance(cv_strategy, GroupKFold) and not self.calibrate:
            cv_param = list(cv_strategy.split(X, y, groups=self._season_groups))
        else:
            cv_param = n_folds  # plain integer; safe to reuse inside CalibratedClassifierCV

        self.model = LogisticRegressionCV(
            Cs=np.logspace(-6, 4, LOGREG_CONFIG['Cs']),
            cv=cv_param,
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
        Fit gradient boosting with optional Optuna hyperparameter tuning.

        When OPTUNA_CONFIG['enabled'] is True, runs a short Optuna study
        (n_trials / timeout from config) using temporal GroupKFold CV to find
        the best regularization / depth / learning-rate combination for the
        current training window.  Because the optimal hyperparameters can shift
        between eras (one-and-done vs. transfer-portal), per-fit tuning is
        more robust than a single hardcoded configuration.

        Sample weights combine:
        1. Class balancing — upweights the rare champion class (~1:87)
        2. Era weighting (optional) — upweights recent seasons
        """
        print(f"Fitting Gradient Boosting (n={len(y)}, positives={int(y.sum())})")

        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        sample_weight = np.where(y == 1, scale_pos_weight, 1.0)
        if self._era_weights is not None:
            sample_weight = sample_weight * self._era_weights
            print(f"  Applied era weights "
                  f"(range: {self._era_weights.min():.2f}–{self._era_weights.max():.2f})")

        # ---- Optuna tuning ---------------------------------------------------
        best_params = self._tune_gbm_optuna(X, y, sample_weight)

        # ---- Final fit with best params on full training set -----------------
        self.model = HistGradientBoostingClassifier(
            max_iter=best_params['max_iter'],
            max_depth=best_params['max_depth'],
            min_samples_leaf=best_params['min_samples_leaf'],
            l2_regularization=best_params['l2_regularization'],
            learning_rate=best_params['learning_rate'],
            early_stopping=GBM_CONFIG['early_stopping'],
            validation_fraction=GBM_CONFIG['validation_fraction'],
            n_iter_no_change=GBM_CONFIG['n_iter_no_change'],
            random_state=GBM_CONFIG['random_state'],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y, sample_weight=sample_weight)

        print(f"  Iterations used: {self.model.n_iter_}")

    def _tune_gbm_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run an Optuna study to find optimal GBM hyperparameters.

        Uses temporal GroupKFold (seasons as groups) so no future-season data
        contaminates validation during tuning.  Falls back to GBM_CONFIG
        defaults if Optuna is disabled or tuning fails.

        Returns
        -------
        dict  Best hyperparameter values (keys match HistGradientBoosting args).
        """
        if not OPTUNA_CONFIG.get('enabled', True):
            return {
                'max_iter':          GBM_CONFIG['max_iter'],
                'max_depth':         GBM_CONFIG['max_depth'],
                'min_samples_leaf':  GBM_CONFIG['min_samples_leaf'],
                'l2_regularization': GBM_CONFIG['l2_regularization'],
                'learning_rate':     GBM_CONFIG['learning_rate'],
            }

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("  [Optuna] Not installed — using default GBM hyperparameters.")
            return {k: GBM_CONFIG[k] for k in
                    ('max_iter', 'max_depth', 'min_samples_leaf',
                     'l2_regularization', 'learning_rate')}

        groups = self._season_groups
        n_folds = OPTUNA_CONFIG.get('cv_folds', 4)
        if groups is not None and len(np.unique(groups)) >= 3:
            cv = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
            splits = list(cv.split(X, y, groups=groups))
        else:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=GBM_CONFIG['random_state'])
            splits = list(skf.split(X, y))

        def objective(trial: 'optuna.Trial') -> float:
            params = {
                'max_iter':          trial.suggest_int(
                    'max_iter', *OPTUNA_CONFIG['max_iter']),
                'max_depth':         trial.suggest_int(
                    'max_depth', *OPTUNA_CONFIG['max_depth']),
                'min_samples_leaf':  trial.suggest_int(
                    'min_samples_leaf', *OPTUNA_CONFIG['min_samples_leaf']),
                'l2_regularization': trial.suggest_float(
                    'l2_regularization', *OPTUNA_CONFIG['l2_regularization'], log=True),
                'learning_rate':     trial.suggest_float(
                    'learning_rate', *OPTUNA_CONFIG['learning_rate'], log=True),
            }

            fold_scores = []
            for tr_idx, val_idx in splits:
                clf = HistGradientBoostingClassifier(
                    **params,
                    early_stopping=False,
                    random_state=GBM_CONFIG['random_state'],
                )
                sw_tr = sample_weight[tr_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X[tr_idx], y[tr_idx], sample_weight=sw_tr)

                from sklearn.metrics import brier_score_loss
                prob = clf.predict_proba(X[val_idx])[:, 1]
                fold_scores.append(brier_score_loss(y[val_idx], prob))

            return float(np.mean(fold_scores))

        try:
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(
                    seed=GBM_CONFIG['random_state'], n_startup_trials=10),
            )
            study.optimize(
                objective,
                n_trials=OPTUNA_CONFIG.get('n_trials', 40),
                timeout=OPTUNA_CONFIG.get('timeout', 60),
                show_progress_bar=False,
            )
            best = study.best_params
            print(f"  [Optuna] Best Brier={study.best_value:.4f}  "
                  f"lr={best['learning_rate']:.4f}  "
                  f"depth={best['max_depth']}  "
                  f"l2={best['l2_regularization']:.2f}")
            return best
        except Exception as exc:
            print(f"  [Optuna] Tuning failed ({exc}) — using defaults.")
            return {k: GBM_CONFIG[k] for k in
                    ('max_iter', 'max_depth', 'min_samples_leaf',
                     'l2_regularization', 'learning_rate')}
        
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Apply probability calibration using Platt Scaling (sigmoid).
        
        NOTE: We use 'sigmoid' instead of 'isotonic' because:
        - Isotonic regression requires hundreds of samples per class for stability
        - With only ~40 champions in our dataset, isotonic overfits severely
        - Sigmoid (Platt Scaling) fits only 2 parameters (slope + intercept),
          making it much more robust for rare event prediction
        
        Reference: Niculescu-Mizil & Caruana (2005) demonstrate sigmoid
        outperforms isotonic on small datasets.
        """
        print("Applying probability calibration (sigmoid/Platt scaling)...")
        
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.model,
            method='sigmoid',  # CRITICAL: sigmoid >> isotonic for rare events
            cv=min(5, int(y.sum()))  # Can use more folds with sigmoid's stability
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
    
    def get_feature_importance(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Get feature importance for any model type.
        
        For logistic regression: uses absolute coefficients
        For GBM with feature_importances_: uses built-in importance
        For HistGBM without feature_importances_: uses permutation importance (requires X, y)
        
        Args:
            X: Optional feature matrix for permutation importance
            y: Optional labels for permutation importance
        
        Returns:
            DataFrame with feature importance
        """
        from sklearn.inspection import permutation_importance
        
        # Determine number of features and names
        if hasattr(self.model, 'coef_'):
            n_features = self.model.coef_.shape[1]
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            n_features = len(importance)
        elif X is not None and y is not None:
            # Use permutation importance for HistGradientBoosting
            print("Computing permutation importance (may take a moment)...")
            perm_result = permutation_importance(
                self.model, X, y, 
                n_repeats=10, 
                random_state=42,
                scoring='neg_log_loss'
            )
            importance = perm_result.importances_mean
            n_features = len(importance)
        elif hasattr(self.model, 'n_features_in_'):
            # Fallback to uniform weights if no data provided
            n_features = self.model.n_features_in_
            print("Warning: Feature importances not available without data. "
                  "Pass X and y to compute permutation importance.")
            importance = np.ones(n_features) / n_features
        else:
            raise ValueError("Model doesn't have interpretable importance")
        
        if self.feature_names is None:
            names = [f'feature_{i}' for i in range(n_features)]
        else:
            names = self.feature_names
            
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
    
    def predict_proba_normalized(
        self, 
        X: np.ndarray, 
        season_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict probabilities with sum-to-one normalization within each season.
        
        The tournament has a constraint: exactly one team wins per season.
        Therefore, probabilities should sum to 1.0 within each season.
        This normalization often improves Brier Score significantly because
        it injects ground-truth domain knowledge into the prediction.
        
        Args:
            X: Feature matrix
            season_ids: Array indicating which season each row belongs to.
                       If None, assumes all rows are from the same season.
                       
        Returns:
            Normalized probabilities that sum to 1.0 within each season
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Get raw probabilities
        probs = self.predict_proba(X)
        
        if season_ids is None:
            # All from same season - simple normalization
            total = probs.sum()
            if total > 0:
                return probs / total
            return probs
        
        # Normalize within each season
        normalized = probs.copy()
        unique_seasons = np.unique(season_ids)
        
        for season in unique_seasons:
            mask = season_ids == season
            season_total = probs[mask].sum()
            if season_total > 0:
                normalized[mask] = probs[mask] / season_total
                
        return normalized


def compute_era_weights(season_years: np.ndarray, decay: float = 0.03) -> np.ndarray:
    """
    Compute per-sample weights that prioritize recent seasons.
    
    Uses exponential decay from the most recent season so that
    the 2024 season has weight 1.0, 2023 has ~0.97, 2014 has ~0.74,
    and 2002 has ~0.51. This addresses concept drift (3-point revolution,
    transfer portal, etc.) without discarding valuable historical data.
    
    Args:
        season_years: Array of season years per sample
        decay: Decay rate per year (default 0.03 = ~3% per year)
    
    Returns:
        Array of weights (0-1), most recent season = 1.0
    """
    max_year = season_years.max()
    years_ago = max_year - season_years
    return np.exp(-decay * years_ago)


def detect_cinderella_profile(
    talent: np.ndarray,
    experience: np.ndarray,
    efficiency: np.ndarray,
    talent_threshold_pct: float = 40.0,
    exp_threshold: float = 2.0,
    em_threshold_pct: float = 70.0
) -> np.ndarray:
    """
    Identify teams with the "Cinderella" profile: Low Talent / High Experience / High Efficiency.
    
    Historical analysis shows mid-major deep runs (George Mason 2006, VCU 2011,
    Butler 2010) share this profile. The model benefits from a binary flag that
    captures this non-linear interaction rather than relying on the model to
    discover it from raw features alone.
    
    Args:
        talent: TALENT scores
        experience: EXP scores
        efficiency: KADJ EM values
        talent_threshold_pct: Percentile below which talent is "low"
        exp_threshold: Minimum EXP for "high experience"
        em_threshold_pct: Percentile above which efficiency is "high"
    
    Returns:
        Binary array (1 = Cinderella profile, 0 = not)
    """
    talent_cutoff = np.nanpercentile(talent, talent_threshold_pct)
    em_cutoff = np.nanpercentile(efficiency, em_threshold_pct)
    
    is_cinderella = (
        (talent <= talent_cutoff) &
        (experience >= exp_threshold) &
        (efficiency >= em_cutoff)
    ).astype(int)
    
    return is_cinderella


class EnsembleChampionPredictor:
    """
    Ensemble predictor that combines Logistic Regression and Gradient Boosting.
    
    Research shows that averaging predictions from multiple model types
    reduces variance significantly, buffering the risks of either model
    failing individually. The linear model captures "resume" features
    while the non-linear model captures interactions.
    
    P_final = weight_lr * P_LR + weight_gbm * P_GBM
    
    Attributes:
        lr_model: Fitted LogisticRegression ChampionPredictor
        gbm_model: Fitted GradientBoosting ChampionPredictor
        weight_lr: Weight for logistic regression predictions (default 0.5)
        weight_gbm: Weight for gradient boosting predictions (default 0.5)
    """
    
    def __init__(
        self,
        weight_lr: float = 0.5,
        weight_gbm: float = 0.5,
        calibrate: bool = True
    ):
        """
        Initialize the ensemble predictor.
        
        Args:
            weight_lr: Weight for logistic regression (0-1)
            weight_gbm: Weight for gradient boosting (0-1)
            calibrate: Whether to apply probability calibration to base models
        """
        if not np.isclose(weight_lr + weight_gbm, 1.0):
            # Normalize weights
            total = weight_lr + weight_gbm
            weight_lr /= total
            weight_gbm /= total
            
        self.weight_lr = weight_lr
        self.weight_gbm = weight_gbm
        self.calibrate = calibrate
        
        self.lr_model = ChampionPredictor(model_type='logreg', calibrate=calibrate)
        self.gbm_model = ChampionPredictor(model_type='gbm', calibrate=calibrate)
        self.feature_names = None
        self._fitted = False
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
        season_groups: Optional[np.ndarray] = None,
        era_weights: Optional[np.ndarray] = None
    ) -> 'EnsembleChampionPredictor':
        """
        Fit both base models on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (1 = champion, 0 = not champion)
            feature_names: Names of features for interpretability
            season_groups: Season year per row for temporal CV
            era_weights: Per-sample weights for concept drift handling
            
        Returns:
            self
        """
        self.feature_names = feature_names
        
        print("="*60)
        print("ENSEMBLE CHAMPION PREDICTOR")
        print(f"Weights: LR={self.weight_lr:.2f}, GBM={self.weight_gbm:.2f}")
        if season_groups is not None:
            print(f"Seasons: {int(np.min(season_groups))}-{int(np.max(season_groups))}")
        print("="*60)
        
        print("\n[1/2] Training Logistic Regression component...")
        self.lr_model.fit(X, y, feature_names=feature_names,
                          season_groups=season_groups, era_weights=era_weights)
        
        print("\n[2/2] Training Gradient Boosting component...")
        self.gbm_model.fit(X, y, feature_names=feature_names,
                          season_groups=season_groups, era_weights=era_weights)
        
        self._fitted = True
        print("\nEnsemble training complete.")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict championship probabilities using weighted ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Weighted average of LR and GBM probabilities
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        lr_probs = self.lr_model.predict_proba(X)
        gbm_probs = self.gbm_model.predict_proba(X)
        
        # Weighted average
        ensemble_probs = self.weight_lr * lr_probs + self.weight_gbm * gbm_probs
        
        return ensemble_probs
    
    def predict_proba_normalized(
        self,
        X: np.ndarray,
        season_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict probabilities with sum-to-one normalization.
        
        Args:
            X: Feature matrix
            season_ids: Array indicating which season each row belongs to.
            
        Returns:
            Normalized ensemble probabilities
        """
        probs = self.predict_proba(X)
        
        if season_ids is None:
            total = probs.sum()
            if total > 0:
                return probs / total
            return probs
        
        normalized = probs.copy()
        unique_seasons = np.unique(season_ids)
        
        for season in unique_seasons:
            mask = season_ids == season
            season_total = probs[mask].sum()
            if season_total > 0:
                normalized[mask] = probs[mask] / season_total
                
        return normalized
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary champion labels."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def get_component_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each component model.
        
        Useful for understanding model disagreement.
        
        Returns:
            Dict with 'lr', 'gbm', and 'ensemble' probability arrays
        """
        lr_probs = self.lr_model.predict_proba(X)
        gbm_probs = self.gbm_model.predict_proba(X)
        ensemble_probs = self.predict_proba(X)
        
        return {
            'lr': lr_probs,
            'gbm': gbm_probs,
            'ensemble': ensemble_probs,
            'disagreement': np.abs(lr_probs - gbm_probs)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get combined feature importance from both models.
        
        Averages the importance from LR (absolute coefficients) 
        and GBM (tree-based importance).
        """
        lr_importance = self.lr_model.get_feature_importance()
        gbm_importance = self.gbm_model.get_feature_importance()
        
        # Normalize each to sum to 1
        lr_importance['importance'] = lr_importance['importance'] / lr_importance['importance'].sum()
        gbm_importance['importance'] = gbm_importance['importance'] / gbm_importance['importance'].sum()
        
        # Merge and average
        merged = lr_importance.merge(
            gbm_importance, 
            on='feature', 
            suffixes=('_lr', '_gbm')
        )
        merged['importance'] = (
            self.weight_lr * merged['importance_lr'] + 
            self.weight_gbm * merged['importance_gbm']
        )
        
        return merged[['feature', 'importance', 'importance_lr', 'importance_gbm']].sort_values(
            'importance', ascending=False
        )


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
    
    # Feature importance (pass data for permutation importance)
    print("\nTop feature importance:")
    importance = gbm.get_feature_importance(X_train, y_train)
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
    
    # Test Ensemble
    print("\n" + "="*60)
    print("--- ENSEMBLE MODEL (LR + GBM) ---")
    print("="*60)
    
    ensemble = EnsembleChampionPredictor(weight_lr=0.5, weight_gbm=0.5, calibrate=False)
    ensemble.fit(X_train, y_train, feature_names=builder.get_feature_names())
    
    # Predict with ensemble
    ensemble_probs = ensemble.predict_proba(X_test)
    test_df_feat['ENSEMBLE_PROB'] = ensemble_probs
    test_df_feat['ENSEMBLE_RANK'] = test_df_feat['ENSEMBLE_PROB'].rank(ascending=False)
    
    print("\nTop 10 by Ensemble:")
    top_10_ens = test_df_feat.nlargest(10, 'ENSEMBLE_PROB')[['TEAM', 'SEED', 'ENSEMBLE_PROB', 'IS_CHAMPION']]
    print(top_10_ens.to_string())
    
    champ_ens = test_df_feat[test_df_feat['IS_CHAMPION'] == 1]
    print(f"\nEnsemble champion rank: {int(champ_ens['ENSEMBLE_RANK'].values[0])}")
    
    # Test normalized probabilities
    print("\n--- Normalized Probabilities (sum-to-one) ---")
    norm_probs = ensemble.predict_proba_normalized(X_test)
    print(f"Sum of raw probabilities: {ensemble_probs.sum():.4f}")
    print(f"Sum of normalized probabilities: {norm_probs.sum():.4f}")
    
    # Show component disagreement
    components = ensemble.get_component_predictions(X_test)
    test_df_feat['DISAGREEMENT'] = components['disagreement']
    print("\nHighest model disagreement (LR vs GBM):")
    high_disagree = test_df_feat.nlargest(5, 'DISAGREEMENT')[['TEAM', 'PROB', 'GBM_PROB', 'DISAGREEMENT']]
    print(high_disagree.to_string())


if __name__ == "__main__":
    main()
