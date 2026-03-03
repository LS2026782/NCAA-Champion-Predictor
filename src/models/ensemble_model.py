"""
Ensemble model: Logistic Regression + Gradient Boosting with dynamic weighting.

Design rationale
----------------
Ensembling a linear model (LogReg) with a non-linear tree model (GBM) provides
meaningful diversity: each component captures different patterns in the data
and their errors are partly uncorrelated.  Averaging two logistic regressions
trained on overlapping feature sets—the previous approach—offers negligible
benefit.

Weighting strategy
------------------
Instead of hardcoded 50/50 weights, each model's contribution is inversely
proportional to its leave-one-season-out Brier score on the training data.
Lower Brier score → lower average error → higher weight.  This gives more
influence to whichever model happens to fit the training periods better, and
adapts automatically when the dataset grows.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.builder import FeatureBuilder
from src.models.champion_model import ChampionPredictor, compute_era_weights


def _brier_cv(model_type: str, X: np.ndarray, y: np.ndarray,
              season_groups: np.ndarray) -> float:
    """
    Estimate leave-one-season-out Brier score via GroupKFold.

    Each fold holds out one season so no future data contaminates the
    cross-validation estimate — the same temporal discipline used in the
    main backtest.
    """
    unique = np.unique(season_groups)
    n_splits = min(5, len(unique))
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, val_idx in gkf.split(X, y, groups=season_groups):
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            if ytr.sum() == 0:
                continue
            era_w = compute_era_weights(season_groups[train_idx])
            m = ChampionPredictor(model_type=model_type, calibrate=False)
            m.fit(Xtr, ytr, season_groups=season_groups[train_idx], era_weights=era_w)
            probs = m.predict_proba(Xval)
            scores.append(brier_score_loss(yval, probs))
    return float(np.mean(scores)) if scores else 0.5


class EnsemblePredictor:
    """
    LogReg + GBM ensemble with Brier-score-based dynamic weighting.

    Both base learners share the same FeatureBuilder (and therefore the same
    feature set and imputation parameters), so the only source of diversity
    is the model class itself: logistic regression captures linear boundaries
    while gradient boosting learns non-linear interactions.

    Attributes
    ----------
    weights : list[float]
        [logreg_weight, gbm_weight] after fitting.  Sum to 1.
    logreg_brier, gbm_brier : float
        CV Brier scores computed during fit, used to derive weights.
    """

    def __init__(self):
        self.builder       = FeatureBuilder()
        self.logreg_model  = ChampionPredictor(model_type='logreg', calibrate=False)
        self.gbm_model     = ChampionPredictor(model_type='gbm',    calibrate=False)
        self.weights: List[float] = [0.5, 0.5]
        self.logreg_brier: float  = 0.5
        self.gbm_brier:    float  = 0.5
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        season_groups: Optional[np.ndarray] = None,
        era_weights:   Optional[np.ndarray] = None,
    ) -> 'EnsemblePredictor':
        """
        Fit both base learners and derive Brier-score-based ensemble weights.

        Parameters
        ----------
        train_df : DataFrame
            Training data (all years before the test year).
        season_groups : array of ints, optional
            Season year per row — used for temporal GroupKFold CV.
        era_weights : array of floats, optional
            Per-sample recency weights (from compute_era_weights).
        """
        _, X_train = self.builder.build_features(train_df, fit_scaler=True)
        y = train_df['IS_CHAMPION'].values

        groups = season_groups if season_groups is not None else (
            train_df['YEAR'].values if 'YEAR' in train_df.columns else None
        )

        # --- 1. Estimate Brier scores via temporal CV (before final fit) -------
        print("  [Ensemble] Computing CV Brier scores to derive weights...")
        if groups is not None and len(np.unique(groups)) >= 3:
            self.logreg_brier = _brier_cv('logreg', X_train, y, groups)
            self.gbm_brier    = _brier_cv('gbm',    X_train, y, groups)
        else:
            self.logreg_brier = self.gbm_brier = 0.25  # equal weight fallback

        # Inverse-Brier weights: better model (lower Brier) gets higher weight.
        # Add small epsilon to avoid division-by-zero if a model is perfect.
        eps = 1e-6
        inv_logreg = 1.0 / (self.logreg_brier + eps)
        inv_gbm    = 1.0 / (self.gbm_brier    + eps)
        total      = inv_logreg + inv_gbm
        self.weights = [inv_logreg / total, inv_gbm / total]

        print(f"  [Ensemble] LogReg Brier={self.logreg_brier:.4f}  "
              f"GBM Brier={self.gbm_brier:.4f}  "
              f"→ weights {self.weights[0]:.2f} / {self.weights[1]:.2f}")

        # --- 2. Final fit on the full training set ----------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.logreg_model.fit(
                X_train, y,
                feature_names=self.builder.get_feature_names(),
                season_groups=groups,
                era_weights=era_weights,
            )
            self.gbm_model.fit(
                X_train, y,
                feature_names=self.builder.get_feature_names(),
                season_groups=groups,
                era_weights=era_weights,
            )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Return weighted-average championship probabilities.

        Parameters
        ----------
        test_df : DataFrame
            Test-year data (unseen; feature builder uses training-set params).
        """
        if not self._fitted:
            raise ValueError("EnsemblePredictor not fitted — call fit() first.")

        _, X_test = self.builder.build_features(test_df, fit_scaler=False)

        probs_lr  = self.logreg_model.predict_proba(X_test)
        probs_gbm = self.gbm_model.predict_proba(X_test)

        return self.weights[0] * probs_lr + self.weights[1] * probs_gbm

    # ------------------------------------------------------------------
    def get_model_contributions(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Return per-team probabilities from each component for inspection."""
        if not self._fitted:
            raise ValueError("EnsemblePredictor not fitted.")

        _, X_test = self.builder.build_features(test_df, fit_scaler=False)

        probs_lr  = self.logreg_model.predict_proba(X_test)
        probs_gbm = self.gbm_model.predict_proba(X_test)
        ensemble  = self.weights[0] * probs_lr + self.weights[1] * probs_gbm

        result = test_df[['TEAM', 'SEED']].copy()
        result['PROB_LOGREG']   = probs_lr
        result['PROB_GBM']      = probs_gbm
        result['PROB_ENSEMBLE'] = ensemble
        for col in ['PROB_LOGREG', 'PROB_GBM', 'PROB_ENSEMBLE']:
            result[col.replace('PROB', 'RANK')] = (
                result[col].rank(ascending=False).astype(int)
            )
        return result.sort_values('PROB_ENSEMBLE', ascending=False)


def backtest_ensemble():
    """Run rolling-year backtest with the LogReg+GBM ensemble."""
    from src.data.loader import DataLoader
    from src.models.champion_model import compute_era_weights
    from config.settings import FIRST_TEST_YEAR

    print("=" * 70)
    print("ENSEMBLE MODEL BACKTEST  (LogReg + GBM, dynamic Brier weights)")
    print("=" * 70)

    loader = DataLoader()
    loader.load_all()

    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    results = []

    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        groups    = train_df['YEAR'].values if 'YEAR' in train_df.columns else None
        era_w     = compute_era_weights(groups) if groups is not None else None

        ensemble = EnsemblePredictor()
        ensemble.fit(train_df, season_groups=groups, era_weights=era_w)

        probs = ensemble.predict_proba(test_df)
        test_df = test_df.copy()
        test_df['PROB'] = probs
        test_df['RANK'] = test_df['PROB'].rank(ascending=False).astype(int)

        champ = test_df[test_df['IS_CHAMPION'] == 1].iloc[0]
        results.append({
            'year': test_year,
            'champion': champ['TEAM'],
            'seed': int(champ['SEED']),
            'rank': int(champ['RANK']),
            'prob': float(champ['PROB']),
            'logreg_weight': ensemble.weights[0],
            'gbm_weight': ensemble.weights[1],
        })

        flag = "Yes" if champ['RANK'] <= 5 else "No"
        print(f"{test_year}: {champ['TEAM']:<22} Seed={int(champ['SEED'])} "
              f"Rank={int(champ['RANK']):2d}  Top5={flag}  "
              f"(w_lr={ensemble.weights[0]:.2f} w_gbm={ensemble.weights[1]:.2f})")

    ranks = [r['rank'] for r in results]
    print("\n" + "=" * 70)
    print("ENSEMBLE SUMMARY")
    print("=" * 70)
    print(f"Mean Champion Rank:   {np.mean(ranks):.2f}")
    print(f"Median Champion Rank: {np.median(ranks):.1f}")
    print(f"Top-1 Rate:  {sum(r == 1 for r in ranks) / len(ranks) * 100:.1f}%")
    print(f"Top-5 Rate:  {sum(r <= 5 for r in ranks) / len(ranks) * 100:.1f}%")
    print(f"Top-10 Rate: {sum(r <= 10 for r in ranks) / len(ranks) * 100:.1f}%")
    return results


if __name__ == "__main__":
    backtest_ensemble()
