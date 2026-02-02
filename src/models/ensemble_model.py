"""
Ensemble model combining Original and Enhanced feature sets.

Strategy: Average the probabilities from both models to get
the best of both worlds.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.builder import FeatureBuilder
from src.features.enhanced_builder import EnhancedFeatureBuilder
from src.models.champion_model import ChampionPredictor


class EnsemblePredictor:
    """
    Ensemble model that combines Original and Enhanced models.
    
    Averaging probabilities from models with different feature sets
    often produces more robust predictions.
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize ensemble.
        
        Args:
            weights: Optional weights for [original, enhanced] models.
                    Default is equal weighting [0.5, 0.5].
        """
        self.weights = weights or [0.5, 0.5]
        
        self.original_builder = FeatureBuilder()
        self.enhanced_builder = EnhancedFeatureBuilder()
        
        self.original_model = ChampionPredictor(model_type='logreg')
        self.enhanced_model = ChampionPredictor(model_type='logreg')
        
        self._fitted = False
        
    def fit(
        self, 
        train_df: pd.DataFrame,
        all_data: Optional[pd.DataFrame] = None
    ) -> 'EnsemblePredictor':
        """
        Fit both models.
        
        Args:
            train_df: Training data
            all_data: Full data for computing ranks (optional)
        """
        # Fit original model
        train_orig, X_orig = self.original_builder.build_features(
            train_df, fit_scaler=True
        )
        y = train_orig['IS_CHAMPION'].values
        self.original_model.fit(
            X_orig, y, 
            feature_names=self.original_builder.get_feature_names()
        )
        
        # Fit enhanced model
        train_enh, X_enh = self.enhanced_builder.build_features(
            train_df, fit_scaler=True, all_data=all_data
        )
        self.enhanced_model.fit(
            X_enh, y,
            feature_names=self.enhanced_builder.get_feature_names()
        )
        
        self._fitted = True
        return self
    
    def predict_proba(
        self, 
        test_df: pd.DataFrame,
        all_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Predict probabilities using ensemble averaging.
        
        Args:
            test_df: Test data
            all_data: Full data for computing ranks
            
        Returns:
            Averaged probability predictions
        """
        if not self._fitted:
            raise ValueError("Model not fitted")
        
        # Get predictions from original model
        test_orig, X_orig = self.original_builder.build_features(
            test_df, fit_scaler=False
        )
        probs_orig = self.original_model.predict_proba(X_orig)
        
        # Get predictions from enhanced model
        test_enh, X_enh = self.enhanced_builder.build_features(
            test_df, fit_scaler=False, all_data=all_data
        )
        probs_enh = self.enhanced_model.predict_proba(X_enh)
        
        # Weighted average
        ensemble_probs = (
            self.weights[0] * probs_orig + 
            self.weights[1] * probs_enh
        )
        
        return ensemble_probs
    
    def get_model_contributions(
        self,
        test_df: pd.DataFrame,
        all_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get individual model predictions for analysis.
        """
        test_orig, X_orig = self.original_builder.build_features(
            test_df.copy(), fit_scaler=False
        )
        probs_orig = self.original_model.predict_proba(X_orig)
        
        test_enh, X_enh = self.enhanced_builder.build_features(
            test_df.copy(), fit_scaler=False, all_data=all_data
        )
        probs_enh = self.enhanced_model.predict_proba(X_enh)
        
        ensemble_probs = (
            self.weights[0] * probs_orig + 
            self.weights[1] * probs_enh
        )
        
        result = test_df[['TEAM', 'SEED']].copy()
        result['PROB_ORIGINAL'] = probs_orig
        result['PROB_ENHANCED'] = probs_enh
        result['PROB_ENSEMBLE'] = ensemble_probs
        result['RANK_ORIGINAL'] = pd.Series(probs_orig).rank(ascending=False).astype(int).values
        result['RANK_ENHANCED'] = pd.Series(probs_enh).rank(ascending=False).astype(int).values
        result['RANK_ENSEMBLE'] = pd.Series(ensemble_probs).rank(ascending=False).astype(int).values
        
        return result.sort_values('PROB_ENSEMBLE', ascending=False)


def backtest_ensemble():
    """Run backtest with ensemble model."""
    from src.data.loader import DataLoader
    from config.settings import FIRST_TEST_YEAR
    
    print("="*70)
    print("ENSEMBLE MODEL BACKTEST")
    print("="*70)
    
    loader = DataLoader()
    loader.load_all()
    full_data = loader.get_data()
    
    years = [y for y in loader.get_years() if y >= FIRST_TEST_YEAR]
    
    results = []
    
    for test_year in years:
        train_df, test_df = loader.get_train_test_split(test_year)
        
        # Fit ensemble
        ensemble = EnsemblePredictor(weights=[0.5, 0.5])
        ensemble.fit(train_df, all_data=full_data)
        
        # Predict
        probs = ensemble.predict_proba(test_df, all_data=full_data)
        
        test_df = test_df.copy()
        test_df['PROB'] = probs
        test_df['RANK'] = test_df['PROB'].rank(ascending=False).astype(int)
        
        champ = test_df[test_df['IS_CHAMPION'] == 1].iloc[0]
        
        results.append({
            'year': test_year,
            'champion': champ['TEAM'],
            'seed': int(champ['SEED']),
            'rank': int(champ['RANK']),
            'prob': champ['PROB']
        })
        
        in_top5 = "Yes" if champ['RANK'] <= 5 else "No"
        print(f"{test_year}: {champ['TEAM']:<20} Seed={int(champ['SEED'])} "
              f"Rank={int(champ['RANK']):2d} Top5={in_top5}")
    
    # Summary
    ranks = [r['rank'] for r in results]
    
    print("\n" + "="*70)
    print("ENSEMBLE SUMMARY")
    print("="*70)
    print(f"Mean Champion Rank:   {np.mean(ranks):.2f}")
    print(f"Median Champion Rank: {np.median(ranks):.1f}")
    print(f"Top-1 Rate:  {sum(r==1 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-5 Rate:  {sum(r<=5 for r in ranks)/len(ranks)*100:.1f}%")
    print(f"Top-10 Rate: {sum(r<=10 for r in ranks)/len(ranks)*100:.1f}%")
    
    return results


if __name__ == "__main__":
    backtest_ensemble()
