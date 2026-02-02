# NCAA Championship Prediction Pipeline

A machine learning pipeline to predict NCAA Men's Basketball Tournament champions using pre-tournament team statistics. The system backtests "champion-likeness" using only data available before Selection Sunday.

## Features

- **Data Pipeline**: Automated loading from Kaggle datasets (KenPom/Barttorvik metrics)
- **Feature Engineering**: Efficiency metrics, Four Factors, experience, strength of schedule
- **Models**: Logistic Regression (interpretable) and Gradient Boosting (ensemble)
- **Backtesting**: Rolling-year cross-validation (train on <Year, test on Year)
- **Monte Carlo Simulation**: Bracket simulation for championship odds
- **Interpretability**: Feature importance and team-level explanations

## Backtest Results (2013-2024)

| Metric | Logistic Regression |
|--------|---------------------|
| Mean Champion Rank | 7.9 |
| Median Champion Rank | 4.0 |
| Top-5 Inclusion Rate | 54.5% |
| Top-10 Inclusion Rate | 72.7% |
| Top-25 Inclusion Rate | 100% |

**Notable Predictions:**
- Correctly predicted Virginia 2019 as #1
- Correctly predicted Kansas 2022 as #1
- All champions ranked in Top-25

## Installation

```bash
# Clone repository
cd "NCAA Champion Predictor"

# Install dependencies
pip install -r requirements.txt

# Setup Kaggle credentials (see below)
```

### Kaggle API Setup

1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to Profile → Settings → API → Create New Token
3. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

## Usage

### Run Full Pipeline
```bash
python main.py
```

### Run Backtest Only
```bash
python main.py --backtest
python main.py --backtest --model gbm  # Use gradient boosting
```

### Predict Specific Year
```bash
python main.py --predict 2024
python main.py --predict 2024 --model logreg
```

### Run Monte Carlo Simulation
```bash
python main.py --simulate 2024 --simulations 10000
```

## Project Structure

```
NCAA Champion Predictor/
├── config/
│   └── settings.py           # Configuration and feature definitions
├── data/
│   ├── raw/                   # Downloaded Kaggle data
│   ├── processed/             # Cleaned datasets
│   └── snapshots/             # Pre-tournament snapshots
├── src/
│   ├── data/
│   │   └── loader.py          # Data loading and preprocessing
│   ├── features/
│   │   └── builder.py         # Feature engineering
│   ├── models/
│   │   └── champion_model.py  # Prediction models
│   ├── evaluation/
│   │   ├── backtester.py      # Rolling-year CV
│   │   └── metrics.py         # Evaluation metrics
│   └── simulation/
│       └── monte_carlo.py     # Bracket simulation
├── results/                   # Output files
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

## Features Used

### Core Efficiency (Most Predictive)
- `KADJ EM` - KenPom Adjusted Efficiency Margin
- `KADJ O` - Adjusted Offensive Efficiency  
- `KADJ D` - Adjusted Defensive Efficiency
- `BARTHAG` - Barttorvik's win probability metric

### Four Factors
- `EFG%` / `EFG%D` - Effective Field Goal %
- `TOV%` / `TOV%D` - Turnover Rate
- `OREB%` / `DREB%` - Rebound Rates
- `FTR` / `FTRD` - Free Throw Rate

### Other
- `EXP` - Team experience
- `ELITE SOS` - Strength of Schedule
- `SEED_STRENGTH` - Derived from tournament seed
- `EM_X_SOS` - Efficiency × SOS interaction

## Leakage Prevention

**Critical**: The pipeline strictly prevents data leakage:
- ✅ Only pre-tournament statistics used as features
- ✅ Tournament results used ONLY for labels (champion identification)
- ✅ Strict temporal splits: train on years < test_year
- ✅ No future data ever leaks into training

## Model Interpretability

### Why a Team Grades as "Champion-Like"

The logistic regression model provides interpretable coefficients:

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| KADJ O | +1.30 | Higher offensive efficiency → more champion-like |
| EM_X_SOS | +1.28 | High efficiency vs tough schedule → more champion-like |
| DREB% | -1.21 | Lower defensive rebounding correlates (counterintuitive - warrants investigation) |
| EFG%D | -1.02 | Better defensive shooting % → more champion-like |

### Team-Level Explanations

```python
from src.models.champion_model import ChampionPredictor
# After fitting model...
model.explain_team(team_features, "Connecticut")
```

## Evaluation Metrics

- **Champion Rank**: Where actual champion falls in probability ranking
- **Top-K Inclusion**: % of years champion was in top K predictions
- **Brier Score**: Probability calibration (lower is better)
- **Log Loss**: Penalizes confident wrong predictions

## Limitations & Caveats

1. **Tournament Randomness**: Single-elimination format has inherent variance. Even a perfect model would miss ~30% of champions.

2. **Sample Size**: Only 16 champions (2008-2024, excluding 2020). Small positive class limits model complexity.

3. **Cinderella Runs**: Low-seeded champions (2014 UConn as 7-seed) are hard to predict.

4. **Injuries**: Pre-tournament stats don't capture mid-season injuries.

## Data Sources

- **Primary**: [Kaggle March Madness Data](https://www.kaggle.com/datasets/nishaanamin/march-madness-data) - KenPom/Barttorvik metrics
- **Secondary**: [538 Team Ratings](https://www.kaggle.com/datasets/raddar/ncaa-men-538-team-ratings)

## License

MIT License - See LICENSE file for details.

## Contributing

Pull requests welcome! Please ensure:
- No data leakage in new features
- Tests pass with `python -m pytest`
- Code follows existing style
