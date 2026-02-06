# NCAA Champion Predictor - Rules Reference

This document consolidates all Cursor rules for the project. The original rules are stored in `.cursor/rules/` and are automatically applied during development.

---

## 1. Project Standards (`project-standards.mdc`)

### Project Structure
- `config/settings.py` - All configuration, feature lists, and constants
- `src/data/` - Data loading and preprocessing
- `src/features/` - Feature engineering pipelines
- `src/models/` - Model definitions (LogReg, GBM, Ensemble)
- `src/evaluation/` - Backtesting and metrics
- `src/simulation/` - Monte Carlo bracket simulation
- `analysis/` - Exploratory analysis scripts

### Key Conventions
1. **Use settings.py for constants** - Never hardcode feature names, paths, or thresholds
2. **Year range**: Data available 2008-2024 (no 2020 due to COVID cancellation)
3. **Entry point**: `main.py` handles CLI and orchestration

### Feature Column Names
Use exact column names from KenPom/Barttorvik data:
- Efficiency: `KADJ EM`, `KADJ O`, `KADJ D`, `BARTHAG`
- Four Factors: `EFG%`, `TOV%`, `OREB%`, `FTR`, `EFG%D`, `TOV%D`, `DREB%`, `FTRD`
- Other: `EXP`, `TALENT`, `ELITE SOS`, `SEED`

---

## 2. Data Leakage Prevention (`data-leakage-prevention.mdc`)

**This is the most critical rule for model integrity.**

### Temporal Split Rules
```python
# CORRECT: Train on past, test on future
train_data = data[data['YEAR'] < test_year]
test_data = data[data['YEAR'] == test_year]

# WRONG: Any overlap or future data in training
train_data = data[data['YEAR'] <= test_year]  # Leaks test year!
```

### Feature Rules

**Allowed (Pre-Tournament Only):**
- Regular season statistics (KenPom, Barttorvik)
- Team seed (announced on Selection Sunday)
- Experience, talent, SOS metrics

**FORBIDDEN (Tournament Results):**
- `POSTSEASON` column - only for labels
- `ROUND` column - only to identify champions
- Any derived feature from tournament performance

### Validation Checklist
Before adding any feature, ask:
1. Was this known BEFORE the tournament started?
2. Could a bettor have accessed this on Selection Sunday?

If NO to either -> **DO NOT USE AS A FEATURE**

---

## 3. Feature Engineering (`feature-engineering.mdc`)

### Feature Categories
Always import feature lists from config:
```python
from config.settings import (
    EFFICIENCY_FEATURES,    # KADJ EM, KADJ O, KADJ D, BARTHAG, etc.
    FOUR_FACTORS_OFF,       # EFG%, TOV%, OREB%, FTR
    FOUR_FACTORS_DEF,       # EFG%D, TOV%D, DREB%, FTRD
    MODEL_FEATURES,         # Final feature set for models
)
```

### Derived Features
```python
# SEED_STRENGTH: Higher = better seed (1-seed gets 16, 16-seed gets 1)
df['SEED_STRENGTH'] = 17 - df['SEED']

# EM_X_SOS: Efficiency x Schedule Strength interaction
df['EM_X_SOS'] = df['KADJ EM'] * df['ELITE SOS']
```

### Scaling Requirements
```python
# Fit scaler on TRAINING data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only!
```

---

## 4. Model Development (`model-development.mdc`)

### Supported Models
1. **Logistic Regression** (default) - Interpretable, stable with small data
2. **Gradient Boosting** - Higher capacity, requires more regularization
3. **Ensemble** - Combines multiple models

### Key Features
The most predictive features for game/champion prediction:
- **KADJ EM** - KenPom efficiency margin (most important)
- **BARTHAG** - Win probability metric
- **TALENT** - Recruiting composite (roster depth proxy, +1.1% accuracy boost)
- **SEED** - Tournament seeding

Note: TALENT (recruiting rankings) is more predictive than EXP (experience).
Champions average 77 TALENT vs 42 for the field.

### Class Imbalance Handling
Only ~1 champion per 68 tournament teams per year:
```python
# Use balanced class weights
LogisticRegressionCV(class_weight='balanced')

# For GBM, consider sample weights
sample_weight = compute_sample_weight('balanced', y_train)
```

### Probability Calibration
- Use `method='sigmoid'` for rare event calibration (better than isotonic)
- Apply sum-to-one normalization for championship probabilities

---

## 5. Backtesting Protocol (`backtesting.mdc`)

### Rolling-Year Cross-Validation
```python
from config.settings import FIRST_TEST_YEAR, MAX_YEAR

for test_year in range(FIRST_TEST_YEAR, MAX_YEAR + 1):
    if test_year == 2020:
        continue  # Skip 2020 (no tournament)
        
    # Strict temporal split
    train = data[data['YEAR'] < test_year]
    test = data[data['YEAR'] == test_year]
```

### Required Metrics
```python
results = {
    'year': test_year,
    'champion': actual_champion,
    'champion_rank': rank_of_actual_champion,
    'top_1_correct': champion_rank == 1,
    'top_5_inclusion': champion_rank <= 5,
    'top_10_inclusion': champion_rank <= 10,
    'predicted_winner': team_with_highest_prob,
}
```

### Aggregate Summary
- Mean/median champion rank
- Top-5, Top-10, Top-25 inclusion rates
- Paired comparison when testing new models

---

## 6. Python Conventions (`python-conventions.mdc`)

### Import Order
```python
# Standard library
from pathlib import Path
import logging

# Third-party
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

# Local imports
from config.settings import MODEL_FEATURES, DATA_DIR
```

### Type Hints & Docstrings
```python
def calculate_champion_probability(
    features: pd.DataFrame,
    model: LogisticRegressionCV
) -> pd.Series:
    """
    Calculate championship probability for each team.
    
    Args:
        features: Pre-tournament team statistics
        model: Fitted classification model
        
    Returns:
        Series with team names as index, probabilities as values
    """
```

### Pandas Best Practices
```python
# Use .loc for explicit indexing
df.loc[df['YEAR'] == 2024, 'SEED_STRENGTH'] = 17 - df['SEED']

# Use .copy() when subsetting
train_df = full_df[full_df['YEAR'] < test_year].copy()
```

### Logging Over Print
```python
import logging
logger = logging.getLogger(__name__)

# Use logging in library code
logger.info(f"Training on {len(train_df)} samples")
logger.warning(f"Missing features: {missing}")

# Print only in scripts/main.py
```

---

## Quick Reference Card

| Rule | Key Point |
|------|-----------|
| Data Leakage | Train < test_year, never <= |
| Features | Only pre-tournament data allowed |
| Champion Label | ROUND == 1 or POSTSEASON == 'Champions' |
| Scaling | Fit on train, transform on test |
| Metrics | Champion rank, Top-K inclusion rates |
| Class Balance | Always use balanced weights |
| Calibration | Use sigmoid, not isotonic |
| Key Features | KADJ EM, BARTHAG, TALENT, SEED |
