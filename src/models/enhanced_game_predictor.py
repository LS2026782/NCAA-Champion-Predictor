"""
Enhanced game predictor with additional features.

This module extends the base GamePredictor with:
1. Neutral court statistics
2. Distance to game location  
3. Historical seed matchup priors
4. Ensemble of multiple models

The goal is to improve upon the baseline 71.1% accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.game_predictor import GamePredictor, GAME_FEATURES
from src.features.game_features import GameFeatureBuilder


# Extended feature set for enhanced predictor
ENHANCED_FEATURES = GAME_FEATURES + [
    # Neutral court features (differences)
    'NEUTRAL_EM_DIFF',
    'NEUTRAL_O_DIFF', 
    'NEUTRAL_D_DIFF',
    # Distance feature
    'DISTANCE_DIFF',
    # Seed prior
    'SEED_PRIOR',
]


class EnhancedGamePredictor:
    """
    Enhanced game predictor with additional features and ensemble option.
    
    Improvements over base predictor:
    1. Uses neutral court stats (tournament games are neutral)
    2. Incorporates distance to game (travel advantage)
    3. Uses historical seed matchup priors
    4. Can ensemble multiple models
    """
    
    def __init__(
        self,
        use_neutral_stats: bool = True,
        use_distance: bool = True,
        use_seed_prior: bool = True,
        use_ensemble: bool = False,
    ):
        """
        Initialize enhanced predictor.
        
        Args:
            use_neutral_stats: Include neutral court performance
            use_distance: Include distance-to-game feature
            use_seed_prior: Include historical seed matchup prior
            use_ensemble: Use ensemble of logreg + gbm
        """
        self.use_neutral_stats = use_neutral_stats
        self.use_distance = use_distance
        self.use_seed_prior = use_seed_prior
        self.use_ensemble = use_ensemble
        
        self.feature_builder = GameFeatureBuilder()
        self.base_features = GAME_FEATURES.copy()
        self.enhanced_features = []
        
        # Build feature list based on options
        if use_neutral_stats:
            self.enhanced_features.extend([
                'NEUTRAL_EM_DIFF', 'NEUTRAL_O_DIFF', 'NEUTRAL_D_DIFF'
            ])
        if use_distance:
            self.enhanced_features.append('DISTANCE_DIFF')
        if use_seed_prior:
            self.enhanced_features.append('SEED_PRIOR')
            
        self.all_features = self.base_features + self.enhanced_features
        
        self.model = None
        self.scaler = StandardScaler()
        self._fitted = False
        
    def fit(
        self,
        games_df: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> 'EnhancedGamePredictor':
        """
        Fit the enhanced model on historical games.
        
        Args:
            games_df: DataFrame with game data
            team_stats: DataFrame with team statistics
            
        Returns:
            self
        """
        self.feature_builder.load_supplementary_data()
        
        # Build feature matrix
        X, y = self._build_training_data(games_df, team_stats)
        
        if len(X) == 0:
            raise ValueError("No valid training games found")
            
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model(s)
        if self.use_ensemble:
            logreg = LogisticRegressionCV(
                Cs=20, cv=5, max_iter=1000, random_state=42
            )
            gbm = HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, min_samples_leaf=20,
                learning_rate=0.05, random_state=42
            )
            self.model = VotingClassifier(
                estimators=[('logreg', logreg), ('gbm', gbm)],
                voting='soft'
            )
        else:
            self.model = LogisticRegressionCV(
                Cs=20, cv=5, max_iter=1000, random_state=42
            )
            
        # Fit
        self.model.fit(X_scaled, y)
        self._fitted = True
        
        print(f"EnhancedGamePredictor fitted on {len(X)} games")
        print(f"  Features: {len(self.all_features)} ({len(self.base_features)} base + {len(self.enhanced_features)} enhanced)")
        
        return self
        
    def _build_training_data(
        self,
        games_df: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build feature matrix from games."""
        X_list = []
        y_list = []
        
        # Index team stats for fast lookup
        stats_indexed = team_stats.set_index(['YEAR', 'TEAM'])
        
        # Also index neutral stats if available
        neutral_indexed = None
        if self.feature_builder.neutral_stats is not None:
            neutral_indexed = self.feature_builder.neutral_stats.set_index(['YEAR', 'TEAM'])
        
        for _, game in games_df.iterrows():
            year = game['YEAR']
            team_a = game['TEAM_A']
            team_b = game['TEAM_B']
            seed_a = game['SEED_A']
            seed_b = game['SEED_B']
            round_num = game['ROUND']
            winner = game['WINNER']
            
            try:
                stats_a = stats_indexed.loc[(year, team_a)]
                stats_b = stats_indexed.loc[(year, team_b)]
            except KeyError:
                continue
                
            # Base features (differences)
            features = []
            for col in self.base_features:
                val_a = stats_a.get(col, 0) if col in stats_a.index else 0
                val_b = stats_b.get(col, 0) if col in stats_b.index else 0
                if pd.isna(val_a): val_a = 0
                if pd.isna(val_b): val_b = 0
                features.append(val_a - val_b)
                
            # Enhanced features
            if self.use_neutral_stats and neutral_indexed is not None:
                try:
                    neutral_a = neutral_indexed.loc[(year, team_a)]
                    neutral_b = neutral_indexed.loc[(year, team_b)]
                    features.append(neutral_a.get('BADJ EM', 0) - neutral_b.get('BADJ EM', 0))
                    features.append(neutral_a.get('BADJ O', 0) - neutral_b.get('BADJ O', 0))
                    features.append(neutral_a.get('BADJ D', 0) - neutral_b.get('BADJ D', 0))
                except KeyError:
                    features.extend([0, 0, 0])
            elif self.use_neutral_stats:
                features.extend([0, 0, 0])
                    
            if self.use_distance:
                dist_features = self.feature_builder.get_distance_features(year, team_a, round_num)
                dist_a = dist_features.get('DISTANCE_MI', 500)
                dist_features = self.feature_builder.get_distance_features(year, team_b, round_num)
                dist_b = dist_features.get('DISTANCE_MI', 500)
                features.append(dist_b - dist_a)  # Positive = A is closer
                
            if self.use_seed_prior:
                prior = self.feature_builder.get_seed_prior(int(seed_a), int(seed_b))
                features.append(prior)
                
            X_list.append(features)
            y_list.append(1 if winner == team_a else 0)
            
        return np.array(X_list), np.array(y_list)
        
    def predict_games(
        self,
        games_df: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict outcomes for multiple games.
        
        Args:
            games_df: DataFrame with game data
            team_stats: DataFrame with team statistics
            
        Returns:
            DataFrame with predictions added
        """
        if not self._fitted:
            raise ValueError("Model not fitted")
        
        self.feature_builder.load_supplementary_data()
        
        results = games_df.copy()
        probs = []
        preds = []
        
        stats_indexed = team_stats.set_index(['YEAR', 'TEAM'])
        neutral_indexed = None
        if self.feature_builder.neutral_stats is not None:
            neutral_indexed = self.feature_builder.neutral_stats.set_index(['YEAR', 'TEAM'])
        
        for _, game in games_df.iterrows():
            year = game['YEAR']
            team_a = game['TEAM_A']
            team_b = game['TEAM_B']
            seed_a = game['SEED_A']
            seed_b = game['SEED_B']
            round_num = game['ROUND']
            
            try:
                stats_a = stats_indexed.loc[(year, team_a)]
                stats_b = stats_indexed.loc[(year, team_b)]
            except KeyError:
                # Default to seed-based prediction if team not found
                probs.append(0.5 + 0.03 * (seed_b - seed_a))
                preds.append(team_a if seed_a <= seed_b else team_b)
                continue
                
            # Build features
            features = []
            for col in self.base_features:
                val_a = stats_a.get(col, 0) if col in stats_a.index else 0
                val_b = stats_b.get(col, 0) if col in stats_b.index else 0
                if pd.isna(val_a): val_a = 0
                if pd.isna(val_b): val_b = 0
                features.append(val_a - val_b)
                
            # Enhanced features
            if self.use_neutral_stats:
                if neutral_indexed is not None:
                    try:
                        neutral_a = neutral_indexed.loc[(year, team_a)]
                        neutral_b = neutral_indexed.loc[(year, team_b)]
                        features.append(neutral_a.get('BADJ EM', 0) - neutral_b.get('BADJ EM', 0))
                        features.append(neutral_a.get('BADJ O', 0) - neutral_b.get('BADJ O', 0))
                        features.append(neutral_a.get('BADJ D', 0) - neutral_b.get('BADJ D', 0))
                    except KeyError:
                        features.extend([0, 0, 0])
                else:
                    features.extend([0, 0, 0])
                    
            if self.use_distance:
                dist_a = self.feature_builder.get_distance_features(year, team_a, round_num)
                dist_b = self.feature_builder.get_distance_features(year, team_b, round_num)
                features.append(dist_b.get('DISTANCE_MI', 500) - dist_a.get('DISTANCE_MI', 500))
                
            if self.use_seed_prior:
                prior = self.feature_builder.get_seed_prior(int(seed_a), int(seed_b))
                features.append(prior)
            
            # Scale and predict
            X = np.array([features])
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = self.scaler.transform(X)
            prob_a = self.model.predict_proba(X_scaled)[0][1]
            
            probs.append(prob_a)
            preds.append(team_a if prob_a > 0.5 else team_b)
        
        results['PROB_A'] = probs
        results['PREDICTED_WINNER'] = preds
        
        return results
        
    def evaluate(
        self,
        games_df: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> Dict:
        """Evaluate model on a set of games."""
        predictions = self.predict_games(games_df, team_stats)
        
        actual = games_df['WINNER'].values
        predicted = predictions['PREDICTED_WINNER'].values
        probs = predictions['PROB_A'].values
        
        y_true = (actual == games_df['TEAM_A'].values).astype(int)
        
        accuracy = accuracy_score(y_true, probs > 0.5)
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        logloss = log_loss(y_true, probs_clipped)
        
        return {
            'accuracy': accuracy,
            'log_loss': logloss,
            'total_games': len(games_df),
            'correct': int((predicted == actual).sum()),
        }


def run_enhanced_backtest(
    use_neutral: bool = True,
    use_distance: bool = True,
    use_seed_prior: bool = True,
    use_ensemble: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run backtest with enhanced predictor.
    
    Args:
        use_neutral: Include neutral court stats
        use_distance: Include distance feature
        use_seed_prior: Include seed priors
        use_ensemble: Use ensemble model
        verbose: Print progress
        
    Returns:
        DataFrame with results
    """
    from src.data.game_loader import GameLoader
    
    # Load data
    loader = GameLoader()
    games_df, team_stats = loader.load_all()
    
    years = sorted(games_df['YEAR'].unique())
    first_test_year = 2012  # Need enough training data
    
    if verbose:
        print("="*70)
        print("ENHANCED GAME PREDICTOR BACKTEST")
        print("="*70)
        features_used = []
        if use_neutral: features_used.append("Neutral Court")
        if use_distance: features_used.append("Distance")
        if use_seed_prior: features_used.append("Seed Prior")
        if use_ensemble: features_used.append("Ensemble")
        print(f"Enhanced features: {', '.join(features_used)}")
        print()
    
    results = []
    
    for test_year in years:
        if test_year < first_test_year or test_year == 2020:
            continue
            
        train_games, test_games = loader.get_train_test_split(test_year)
        
        if verbose:
            print(f"Testing {test_year}...", end=" ")
            
        # Create and fit predictor
        predictor = EnhancedGamePredictor(
            use_neutral_stats=use_neutral,
            use_distance=use_distance,
            use_seed_prior=use_seed_prior,
            use_ensemble=use_ensemble
        )
        
        try:
            predictor.fit(train_games, team_stats)
            metrics = predictor.evaluate(test_games, team_stats)
            
            results.append({
                'year': test_year,
                **metrics
            })
            
            if verbose:
                print(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total_games']})")
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
                
    results_df = pd.DataFrame(results)
    
    if verbose and len(results_df) > 0:
        total_games = results_df['total_games'].sum()
        total_correct = results_df['correct'].sum()
        print()
        print("="*70)
        print(f"OVERALL: {total_correct/total_games:.1%} ({total_correct}/{total_games} games)")
        print(f"Mean Log Loss: {results_df['log_loss'].mean():.4f}")
        print("="*70)
        
    return results_df


def compare_configurations():
    """Compare different feature configurations."""
    print("\n" + "="*70)
    print("COMPARING CONFIGURATIONS")
    print("="*70 + "\n")
    
    configs = [
        ("Baseline (no enhancements)", False, False, False, False),
        ("+ Neutral Court", True, False, False, False),
        ("+ Distance", False, True, False, False),
        ("+ Seed Prior", False, False, True, False),
        ("Neutral + Distance", True, True, False, False),
        ("Neutral + Seed Prior", True, False, True, False),
        ("All Features", True, True, True, False),
        ("All Features + Ensemble", True, True, True, True),
    ]
    
    comparison = []
    
    for name, neutral, distance, seed_prior, ensemble in configs:
        print(f"\nTesting: {name}")
        results = run_enhanced_backtest(
            use_neutral=neutral,
            use_distance=distance,
            use_seed_prior=seed_prior,
            use_ensemble=ensemble,
            verbose=False
        )
        
        total = results['total_games'].sum()
        correct = results['correct'].sum()
        accuracy = correct / total
        
        comparison.append({
            'config': name,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'log_loss': results['log_loss'].mean()
        })
        
        print(f"  Accuracy: {accuracy:.1%}")
        
    print("\n" + "="*70)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("="*70)
    
    comparison_df = pd.DataFrame(comparison).sort_values('accuracy', ascending=False)
    for _, row in comparison_df.iterrows():
        print(f"  {row['config']:30s}: {row['accuracy']:.1%} (log_loss: {row['log_loss']:.4f})")
        
    return comparison_df


def main():
    """Run enhanced backtest."""
    # First, run comparison of configurations
    comparison = compare_configurations()
    
    # Save comparison results
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    comparison.to_csv(results_dir / 'enhanced_comparison.csv', index=False)
    

if __name__ == "__main__":
    main()
