"""
Game-by-game prediction model for NCAA Tournament.

This module predicts individual game outcomes based on team statistics.
Unlike the ChampionPredictor which identifies "champion-like" teams,
this model predicts P(Team A beats Team B) for specific matchups.

APPROACH:
- Features: Difference in team statistics (Team A - Team B)
- Target: 1 if Team A won, 0 if Team B won
- Model: Logistic Regression (interpretable) or Gradient Boosting

LEAKAGE PREVENTION:
- Only pre-tournament statistics used as features
- Train on games from years < test_year
- Test on games in test_year
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import LOGREG_CONFIG, GBM_CONFIG


# Features to use for game prediction (differences between teams)
GAME_FEATURES = [
    'KADJ EM',      # Efficiency margin
    'KADJ O',       # Offensive efficiency
    'KADJ D',       # Defensive efficiency
    'BARTHAG',      # Win probability metric
    'EFG%',         # Effective FG%
    'TOV%',         # Turnover rate
    'OREB%',        # Offensive rebounding
    'FTR',          # Free throw rate
    'EFG%D',        # Defensive EFG%
    'TOV%D',        # Forced turnover rate
    'DREB%',        # Defensive rebounding
    'FTRD',         # Opponent FT rate
    'EXP',          # Experience
    'ELITE SOS',    # Strength of schedule
    'SEED_NUM',     # Tournament seed (numeric)
    'TALENT',       # Recruiting talent composite (roster depth proxy)
]


class GamePredictor:
    """
    Predicts outcomes of individual tournament games.
    
    Uses the difference in team statistics to predict which team wins.
    Trained on historical tournament game outcomes.
    
    Attributes:
        model: Fitted classification model
        scaler: StandardScaler for feature normalization
        feature_cols: List of features used for prediction
    """
    
    def __init__(
        self, 
        model_type: str = 'logreg',
        feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize the game predictor.
        
        Args:
            model_type: 'logreg' or 'gbm'
            feature_cols: Features to use (defaults to GAME_FEATURES)
        """
        self.model_type = model_type
        self.feature_cols = feature_cols or GAME_FEATURES
        self.model = None
        self.scaler = StandardScaler()
        self._fitted = False
        
    def fit(
        self, 
        games_df: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> 'GamePredictor':
        """
        Fit the model on historical tournament games.
        
        Args:
            games_df: DataFrame with columns [YEAR, TEAM_A, TEAM_B, WINNER]
            team_stats: DataFrame with team statistics (indexed by YEAR, TEAM)
            
        Returns:
            self
        """
        # Build feature matrix from games
        X, y = self._build_training_data(games_df, team_stats)
        
        if len(X) == 0:
            raise ValueError("No valid training games found")
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.model_type == 'logreg':
            self.model = LogisticRegressionCV(
                Cs=20,
                cv=5,
                penalty='l2',
                scoring='neg_log_loss',
                max_iter=1000,
                random_state=42
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                min_samples_leaf=20,
                l2_regularization=1.0,
                learning_rate=0.05,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42
            )
            
        # Fit model
        self.model.fit(X_scaled, y)
        self._fitted = True
        
        print(f"GamePredictor fitted on {len(X)} games")
        return self
    
    def _build_training_data(
        self,
        games_df: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix from games DataFrame.
        
        Features are the difference: (Team A stats) - (Team B stats)
        Target is 1 if Team A won, 0 if Team B won.
        """
        X_list = []
        y_list = []
        
        # Index team stats for fast lookup
        stats_indexed = team_stats.set_index(['YEAR', 'TEAM'])
        
        for _, game in games_df.iterrows():
            year = game['YEAR']
            team_a = game['TEAM_A']
            team_b = game['TEAM_B']
            winner = game['WINNER']
            
            try:
                stats_a = stats_indexed.loc[(year, team_a)]
                stats_b = stats_indexed.loc[(year, team_b)]
            except KeyError:
                # Team not found in stats, skip this game
                continue
                
            # Compute feature differences (A - B)
            diff = []
            for col in self.feature_cols:
                val_a = stats_a.get(col, 0) if col in stats_a.index else 0
                val_b = stats_b.get(col, 0) if col in stats_b.index else 0
                # Handle NaN
                if pd.isna(val_a):
                    val_a = 0
                if pd.isna(val_b):
                    val_b = 0
                diff.append(val_a - val_b)
                
            X_list.append(diff)
            y_list.append(1 if winner == team_a else 0)
            
        return np.array(X_list), np.array(y_list)
    
    def predict_game(
        self,
        team_a_stats: Dict,
        team_b_stats: Dict
    ) -> Tuple[float, str]:
        """
        Predict the outcome of a single game.
        
        Args:
            team_a_stats: Statistics for team A
            team_b_stats: Statistics for team B
            
        Returns:
            Tuple of (probability Team A wins, predicted winner name)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Build feature difference
        diff = []
        for col in self.feature_cols:
            val_a = team_a_stats.get(col, 0)
            val_b = team_b_stats.get(col, 0)
            if pd.isna(val_a):
                val_a = 0
            if pd.isna(val_b):
                val_b = 0
            diff.append(val_a - val_b)
            
        X = np.array([diff])
        X_scaled = self.scaler.transform(X)
        
        prob_a = self.model.predict_proba(X_scaled)[0][1]
        
        return prob_a, 'A' if prob_a > 0.5 else 'B'
    
    def predict_games(
        self,
        games_df: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict outcomes for multiple games.
        
        Args:
            games_df: DataFrame with [YEAR, TEAM_A, TEAM_B]
            team_stats: DataFrame with team statistics
            
        Returns:
            DataFrame with predictions added
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        results = games_df.copy()
        probs = []
        preds = []
        
        stats_indexed = team_stats.set_index(['YEAR', 'TEAM'])
        
        for _, game in games_df.iterrows():
            year = game['YEAR']
            team_a = game['TEAM_A']
            team_b = game['TEAM_B']
            
            try:
                stats_a = stats_indexed.loc[(year, team_a)].to_dict()
                stats_b = stats_indexed.loc[(year, team_b)].to_dict()
                prob_a, _ = self.predict_game(stats_a, stats_b)
                probs.append(prob_a)
                preds.append(team_a if prob_a > 0.5 else team_b)
            except KeyError:
                probs.append(0.5)
                preds.append(team_a)  # Default to higher seed
                
        results['PROB_A'] = probs
        results['PREDICTED_WINNER'] = preds
        
        return results
    
    def evaluate(
        self,
        games_df: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> Dict:
        """
        Evaluate model on a set of games.
        
        Args:
            games_df: DataFrame with [YEAR, TEAM_A, TEAM_B, WINNER]
            team_stats: Team statistics
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict_games(games_df, team_stats)
        
        # Calculate metrics
        actual = games_df['WINNER'].values
        predicted = predictions['PREDICTED_WINNER'].values
        probs = predictions['PROB_A'].values
        
        # Convert to binary for sklearn
        y_true = (actual == games_df['TEAM_A'].values).astype(int)
        
        accuracy = accuracy_score(y_true, probs > 0.5)
        
        # Clip probabilities to avoid log(0)
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        logloss = log_loss(y_true, probs_clipped)
        brier = brier_score_loss(y_true, probs)
        
        # Upset analysis (lower seed beating higher seed)
        predictions['SEED_A'] = games_df['SEED_A']
        predictions['SEED_B'] = games_df['SEED_B']
        predictions['ACTUAL_WINNER'] = actual
        
        upsets = predictions[predictions['SEED_A'] > predictions['SEED_B']]
        upset_occurred = (upsets['ACTUAL_WINNER'] == upsets['TEAM_A']).sum()
        upset_predicted = (upsets['PREDICTED_WINNER'] == upsets['TEAM_A']).sum()
        
        return {
            'accuracy': accuracy,
            'log_loss': logloss,
            'brier_score': brier,
            'total_games': len(games_df),
            'correct_predictions': int((predicted == actual).sum()),
            'upset_games': len(upsets),
            'upsets_occurred': int(upset_occurred),
            'upsets_predicted': int(upset_predicted),
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the fitted model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self._fitted:
            raise ValueError("Model not fitted")
            
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = np.zeros(len(self.feature_cols))
            
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)


def main():
    """Test game predictor."""
    print("GamePredictor module loaded successfully")
    print(f"Default features: {GAME_FEATURES}")
    

if __name__ == "__main__":
    main()
