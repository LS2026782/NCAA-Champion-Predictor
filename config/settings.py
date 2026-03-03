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

# Training year range: 2002 is the optimal start year per longitudinal analysis.
# KenPom efficiency metrics are reliable from 2002; pre-2002 data lacks
# possession-based stats and introduces concept drift (shot clock changes, etc.)
MIN_YEAR = 2002
MAX_YEAR = 2025  # Last complete tournament year (2024-25 season)
COVID_YEAR = 2020  # No tournament held

# Minimum years of training data before we start testing
MIN_TRAIN_YEARS = 4

# First test year = MIN_YEAR + MIN_TRAIN_YEARS
FIRST_TEST_YEAR = MIN_YEAR + MIN_TRAIN_YEARS  # 2006

# =============================================================================
# RECONSTRUCTION CONSTANTS
# =============================================================================

# BARTHAG Pythagorean exponent (optimal for NCAA possession variance)
BARTHAG_EXPONENT = 11.5

# TALENT reconstruction: exponential decay for RSCI rankings
TALENT_DECAY_SIGMA = 25.0      # Controls how steeply value drops with rank
TALENT_UNRANKED_VALUE = 0.5    # Replacement-level score for unranked players
TALENT_MAX_RANK = 100          # RSCI covers top 100 recruits

# EXPERIENCE class index values
EXP_CLASS_VALUES = {'fr': 0, 'so': 1, 'jr': 2, 'sr': 3, 'gr': 3}

# EFFECTIVE HEIGHT: fraction of minutes defining "interior" players
EFF_HEIGHT_MINUTES_FRACTION = 0.40

# ELITE SOS: opponent rank threshold
ELITE_SOS_RANK_THRESHOLD = 50

# Years requiring feature reconstruction (no native Barttorvik coverage)
RECONSTRUCTION_YEARS = list(range(2002, 2008))

# Era boundaries for concept drift normalization
ERA_BOUNDARIES = {
    'veteran_era': (2002, 2006),     # Experience-dominant champions
    'one_and_done': (2007, 2015),    # Freshman-led dominance begins
    'transfer_portal': (2016, 2026), # Return to veteran value
}

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
    
    # Recruiting talent (roster depth proxy - champions avg 77 vs field 42)
    'TALENT',
    
    # Schedule strength
    'ELITE SOS',
    
    # Derived features (added during feature engineering)
    'SEED_STRENGTH',      # 17 - seed (higher = better seed)
    'EM_X_SOS',           # Efficiency margin * SOS interaction
    
    # Longitudinal features (added for 2002-2025 extended dataset)
    'TALENT_X_EXP',       # Talent-Experience interaction (Cinderella vs Blue Blood)
    'CINDERELLA',         # Binary flag: Low Talent / High EXP / High EM
    'EFG_MARGIN',         # EFG% - EFG%D (shooting differential)
    'RELATIVE_3PR',       # 3-point rate relative to season average (handles 3pt revolution)
]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Logistic Regression hyperparameters
# Wider C grid (-6 to 4) to accommodate the larger 2002-2025 dataset where
# the optimal regularization strength may differ from the 2008-2024 window.
LOGREG_CONFIG = {
    'Cs': 30,                    # Denser grid for finer C selection
    'cv': 5,                     # Cross-validation folds
    'penalty': 'l2',             # Regularization type
    'class_weight': 'balanced',  # Handle class imbalance
    'scoring': 'neg_log_loss',   # Optimize for probability calibration
    'max_iter': 2000,            # Increased for larger dataset convergence
    'random_state': 42,
}

# Gradient Boosting hyperparameters (strict regularization)
# With ~22 positive samples (2002-2024 minus 2020), overfitting is the
# primary risk. Shallow trees + strong L2 + slow learning rate are critical.
GBM_CONFIG = {
    'max_iter': 150,             # Slightly more iterations for larger data
    'max_depth': 3,              # Shallow trees prevent overfitting
    'min_samples_leaf': 20,      # Conservative leaf size
    'l2_regularization': 1.5,    # Stronger regularization for expanded dataset
    'learning_rate': 0.03,       # Slower learning for stability
    'early_stopping': True,
    'validation_fraction': 0.15,
    'n_iter_no_change': 15,      # More patience before stopping
    'random_state': 42,
}

# =============================================================================
# SEED → HISTORICAL CHAMPIONSHIP RATE
# =============================================================================
# Per-seed championship win rate from 1985-2024 (40 tournaments, 4 seeds of
# each number per year = 160 total entries per seed).  These empirical rates
# encode the non-linear relationship between seeding and winning: the gap from
# 1 → 2 is enormous; the gap from 8 → 9 is negligible.
# Used by FeatureBuilder to replace the linear (17 - seed) with a calibrated
# log-probability feature.
SEED_CHAMPION_RATE = {
    1:  0.163,   # 26/160  — 65% of all champions
    2:  0.038,   # 6/160
    3:  0.019,   # 3/160
    4:  0.019,   # 3/160
    5:  0.006,   # 1/160   (UMBC / Villanova '85 era)
    6:  0.006,
    7:  0.006,
    8:  0.006,   # UMBC is the only 16 to beat a 1; 8-seeds have won once
    9:  0.003,
    10: 0.003,
    11: 0.006,   # VCU / George Mason Cinderella surge
    12: 0.002,
    13: 0.001,
    14: 0.001,
    15: 0.001,
    16: 0.001,   # Keep nonzero for numerical stability
}

# =============================================================================
# OPTUNA HYPERPARAMETER TUNING CONFIGURATION
# =============================================================================
OPTUNA_CONFIG = {
    'enabled': True,          # Set False to skip tuning and use GBM_CONFIG defaults
    'n_trials': 40,           # Trials per rolling-year fit (40 is ~30 s on a laptop)
    'timeout': 60,            # Hard wall-clock timeout per fit (seconds)
    'cv_folds': 4,            # Inner CV folds for Optuna objective
    'scoring': 'neg_brier_score',
    'random_state': 42,
    # Search bounds for HistGradientBoostingClassifier
    'max_iter':          (80, 300),
    'max_depth':         (2, 6),
    'min_samples_leaf':  (10, 60),
    'l2_regularization': (0.1, 5.0),
    'learning_rate':     (0.005, 0.15),
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
