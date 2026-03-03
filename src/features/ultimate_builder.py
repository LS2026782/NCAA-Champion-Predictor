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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SKPipeline
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
        # sklearn Pipeline (SimpleImputer → StandardScaler) fitted on training data.
        # Replaced manual self.scaler + self._fit_params['imputation_medians'] tracking.
        self.preprocessor: Optional[ColumnTransformer] = None
        self._fitted = False

        # Composite-feature engineering bounds — stored as named attrs rather
        # than in an opaque dict so type-checkers and readers can see them.
        self._sos_min: Optional[float] = None
        self._sos_max: Optional[float] = None
        self._ft_min:  Optional[float] = None
        self._ft_max:  Optional[float] = None
        self._wab_min: Optional[float] = None
        self._wab_max: Optional[float] = None

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
        """Load preseason and resume data.

        Files are optional — if absent the corresponding features are set to 0
        and a warning is emitted so the gap is visible rather than silent.
        """
        import logging
        _log = logging.getLogger(__name__)

        preseason_path = RAW_DATA_DIR / 'KenPom Preseason.csv'
        if preseason_path.exists():
            try:
                self.preseason_df = pd.read_csv(preseason_path)
            except Exception as exc:
                _log.warning("Could not read %s: %s — MOMENTUM_EM will be 0", preseason_path, exc)
                self.preseason_df = None
        else:
            _log.info("Optional file not found: %s — MOMENTUM_EM will be 0", preseason_path)
            self.preseason_df = None

        resume_path = RAW_DATA_DIR / 'Resumes.csv'
        if resume_path.exists():
            try:
                self.resume_df = pd.read_csv(resume_path)
            except Exception as exc:
                _log.warning("Could not read %s: %s — Q1_WINS will be 0", resume_path, exc)
                self.resume_df = None
        else:
            _log.info("Optional file not found: %s — Q1_WINS will be 0", resume_path)
            self.resume_df = None
    
    def build_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False,
        all_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build ultimate feature matrix."""
        df = df.copy()

        df = self._add_seed_features(df)
        df = self._add_momentum_features(df)
        df = self._add_resume_features(df)
        df = self._add_composite_features(df, fit=fit_scaler)

        missing = [f for f in self.feature_cols if f not in df.columns]
        if missing:
            # Use np.nan instead of 0 so that HistGradientBoostingClassifier
            # can learn native missing-value splits.  Filling with 0 conflates
            # "feature absent" with "feature = 0", which biases tree thresholds
            # downward and degrades split quality on the remaining data.
            print(f"  Missing features (mapped to NaN for native missing splits): {missing}")
            for f in missing:
                df[f] = np.nan

        X = df[self.feature_cols].copy()

        # Preprocessing: sklearn Pipeline replaces the old manual imputation
        # loop and manual StandardScaler calls.  fit_transform() is called
        # only on training data; transform()-only is enforced on test/inference
        # data so no test-set statistics bleed into the pipeline parameters.
        if fit_scaler:
            self._fit_preprocessor(X)
            X_processed = self.preprocessor.transform(X)
            self._fitted = True
        elif self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X.values

        self.feature_names = self.feature_cols
        return df, X_processed

    def _fit_preprocessor(self, X: pd.DataFrame) -> None:
        """
        Build and fit the sklearn ColumnTransformer preprocessing pipeline.

        Columns are split into two groups based on their missingness pattern
        in the training data:

        - Partially missing (some rows NaN, some not): receive median imputation
          followed by standard scaling.  Both statistics are learned exclusively
          from the training data and frozen for inference.
        - Entirely absent (all rows NaN): passed through as-is.  Keeping them as
          NaN lets HistGradientBoostingClassifier route on missingness natively
          rather than conflating absence with a zero value.

        Fully present columns (no NaN) skip the imputer and go straight to the
        scaler — SimpleImputer is a no-op on them and would pass through
        unchanged, so we include them in the same transformer group for
        simplicity.
        """
        absent_cols       = [c for c in X.columns if X[c].isna().all()]
        impute_scale_cols = [c for c in X.columns if c not in absent_cols]

        transformers: list = []
        if impute_scale_cols:
            transformers.append((
                'impute_scale',
                SKPipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler',  StandardScaler()),
                ]),
                impute_scale_cols,
            ))
        if absent_cols:
            # passthrough preserves NaN — native HistGBM missing-value splits
            transformers.append(('passthrough_nan', 'passthrough', absent_cols))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
        )
        self.preprocessor.fit(X)

    def _add_seed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add non-linear seed feature (log of historical championship rate)."""
        from config.settings import SEED_CHAMPION_RATE
        if df['SEED'].dtype == 'object':
            df['SEED_NUM'] = df['SEED'].astype(str).str.extract(r'(\d+)').astype(float)
        else:
            df['SEED_NUM'] = df['SEED'].astype(float)

        rate = df['SEED_NUM'].map(SEED_CHAMPION_RATE).fillna(SEED_CHAMPION_RATE[16])
        df['SEED_STRENGTH'] = np.log(rate)
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
    
    def _add_composite_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Add composite features using training-set normalization to prevent leakage.

        Normalization bounds are stored as typed instance attributes (not in a
        generic dict) so callers and type-checkers have clear visibility of the
        fitted state without the opacity of self._fit_params.
        """

        # EM × SOS interaction — store min/max bounds from training set only
        if 'KADJ EM' in df.columns and 'ELITE SOS' in df.columns:
            if fit:
                self._sos_min = float(df['ELITE SOS'].min())
                self._sos_max = float(df['ELITE SOS'].max())
            sos_min = self._sos_min if self._sos_min is not None else df['ELITE SOS'].min()
            sos_max = self._sos_max if self._sos_max is not None else df['ELITE SOS'].max()
            sos_norm = (df['ELITE SOS'] - sos_min) / (sos_max - sos_min + 1e-8)
            df['EM_X_SOS'] = df['KADJ EM'] * (0.5 + sos_norm)
        else:
            df['EM_X_SOS'] = np.nan

        # CLUTCH_COMPOSITE — store FT% and WAB bounds from training set only
        if 'FT%' in df.columns and 'WAB' in df.columns:
            if fit:
                self._ft_min  = float(df['FT%'].min())
                self._ft_max  = float(df['FT%'].max())
                self._wab_min = float(df['WAB'].min())
                self._wab_max = float(df['WAB'].max())
            ft_min  = self._ft_min  if self._ft_min  is not None else df['FT%'].min()
            ft_max  = self._ft_max  if self._ft_max  is not None else df['FT%'].max()
            wab_min = self._wab_min if self._wab_min is not None else df['WAB'].min()
            wab_max = self._wab_max if self._wab_max is not None else df['WAB'].max()
            ft_norm  = (df['FT%'] - ft_min)  / (ft_max  - ft_min  + 1e-8)
            wab_norm = (df['WAB'] - wab_min) / (wab_max - wab_min + 1e-8)
            df['CLUTCH_COMPOSITE'] = 0.5 * ft_norm + 0.5 * wab_norm
        else:
            df['CLUTCH_COMPOSITE'] = np.nan

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
