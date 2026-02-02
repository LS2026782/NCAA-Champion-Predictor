"""
Configuration settings for NCAA Championship Prediction Pipeline.

This module defines all feature columns, paths, and constants used throughout
the prediction pipeline. Centralizing configuration here prevents magic strings
and makes the pipeline easier to modify.
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# DATA FILES
# =============================================================================

KENPOM_BARTTORVIK_FILE = RAW_DATA_DIR / "KenPom Barttorvik.csv"
TOURNAMENT_MATCHUPS_FILE = RAW_DATA_DIR / "Tournament Matchups.csv"
TEAM_RESULTS_FILE = RAW_DATA_DIR / "Team Results.csv"
SEED_RESULTS_FILE = RAW_DATA_DIR / "Seed Results.csv"

# =============================================================================
# YEAR CONFIGURATION
# =============================================================================

# Training year range (2008 is first year with complete KenPom/Barttorvik data)
MIN_YEAR = 2008
MAX_YEAR = 2024  # Last complete tournament year

# Minimum years of training data before we start testing
MIN_TRAIN_YEARS = 4

# First test year = MIN_YEAR + MIN_TRAIN_YEARS
FIRST_TEST_YEAR = MIN_YEAR + MIN_TRAIN_YEARS  # 2012

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Core efficiency features (adjusted for opponent strength)
EFFICIENCY_FEATURES = [
    'KADJ O',      # KenPom Adjusted Offensive Efficiency
    'KADJ D',      # KenPom Adjusted Defensive Efficiency  
    'KADJ EM',     # KenPom Adjusted Efficiency Margin
    'BADJ EM',     # Barttorvik Adjusted Efficiency Margin
    'BADJ O',      # Barttorvik Adjusted Offensive Efficiency
    'BADJ D',      # Barttorvik Adjusted Defensive Efficiency
    'BARTHAG',     # Barttorvik's win probability metric
]

# Four Factors - Offensive
FOUR_FACTORS_OFF = [
    'EFG%',        # Effective Field Goal % (accounts for 3PT value)
    'TOV%',        # Turnover Rate (lower is better for offense)
    'OREB%',       # Offensive Rebound Rate
    'FTR',         # Free Throw Rate (FTA/FGA)
]

# Four Factors - Defensive
FOUR_FACTORS_DEF = [
    'EFG%D',       # Opponent Effective FG%
    'TOV%D',       # Opponent Turnover Rate (higher is better for defense)
    'DREB%',       # Defensive Rebound Rate
    'FTRD',        # Opponent Free Throw Rate
]

# Experience and talent metrics
EXPERIENCE_FEATURES = [
    'EXP',         # Team experience rating
    'TALENT',      # Recruiting talent composite
]

# Tempo and style
TEMPO_FEATURES = [
    'KADJ T',      # KenPom Adjusted Tempo
    'BADJ T',      # Barttorvik Adjusted Tempo
]

# Schedule strength
SOS_FEATURES = [
    'ELITE SOS',   # Strength of Schedule vs elite opponents
]

# Shooting detail
SHOOTING_FEATURES = [
    '2PT%',        # 2-point FG%
    '3PT%',        # 3-point FG%
    '2PT%D',       # Opponent 2PT%
    '3PT%D',       # Opponent 3PT%
]

# Tournament seeding (only for tournament-qualified teams)
SEED_FEATURE = 'SEED'

# All features combined (base set without derived features)
BASE_FEATURES = (
    EFFICIENCY_FEATURES + 
    FOUR_FACTORS_OFF + 
    FOUR_FACTORS_DEF + 
    EXPERIENCE_FEATURES + 
    TEMPO_FEATURES + 
    SOS_FEATURES +
    SHOOTING_FEATURES
)

# Features we'll actually use in the model (can be tuned)
MODEL_FEATURES = [
    # Primary efficiency (most predictive)
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
    
    # Experience (historically important for tournament success)
    'EXP',
    
    # Schedule strength
    'ELITE SOS',
    
    # Derived features (added during feature engineering)
    'SEED_STRENGTH',      # 17 - seed (higher = better seed)
    'EM_X_SOS',           # Efficiency margin * SOS interaction
]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Logistic Regression hyperparameters
LOGREG_CONFIG = {
    'Cs': 20,                    # Number of regularization strengths to try
    'cv': 5,                     # Cross-validation folds
    'penalty': 'l2',             # Regularization type
    'class_weight': 'balanced',  # Handle class imbalance
    'scoring': 'neg_log_loss',   # Optimize for probability calibration
    'max_iter': 1000,
    'random_state': 42,
}

# Gradient Boosting hyperparameters (strict regularization)
GBM_CONFIG = {
    'max_iter': 100,
    'max_depth': 3,              # Shallow trees prevent overfitting
    'min_samples_leaf': 20,      # Conservative leaf size
    'l2_regularization': 1.0,    # Strong regularization
    'learning_rate': 0.05,       # Slow learning
    'early_stopping': True,
    'validation_fraction': 0.15,
    'n_iter_no_change': 10,
    'random_state': 42,
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Top-K thresholds for inclusion rate metrics
TOP_K_THRESHOLDS = [1, 5, 10, 25, 68]

# Monte Carlo simulation parameters
MONTE_CARLO_SIMULATIONS = 10000

# =============================================================================
# LABEL CONFIGURATION
# =============================================================================

# Round values in KenPom/Barttorvik data represent "teams remaining"
# Lower number = further in tournament = better performance
ROUND_CHAMPION = 1       # Won championship (1 team left)
ROUND_FINAL = 2          # Lost in championship game (2 teams left)
ROUND_FINAL_FOUR = 4     # Lost in Final Four (4 teams left)
ROUND_ELITE_EIGHT = 8    # Lost in Elite Eight (8 teams left)
ROUND_SWEET_SIXTEEN = 16 # Lost in Sweet 16 (16 teams left)
ROUND_ROUND_OF_32 = 32   # Lost in Round of 32 (32 teams left)
ROUND_ROUND_OF_64 = 64   # Lost in Round of 64 (64 teams left)
ROUND_FIRST_FOUR = 68    # Lost in First Four (68 teams = play-in)
