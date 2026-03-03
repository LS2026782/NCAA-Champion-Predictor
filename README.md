# NCAA Championship Prediction Pipeline

A machine learning pipeline for predicting NCAA Men's Basketball Tournament champions using pre-tournament team statistics. The system trains on 23 years of historical data (2002–2025) and backtests champion-likeness using only data available before Selection Sunday.

---

## Backtest Results (2006–2025, 19 seasons)

| Metric | Logistic Regression | Gradient Boosting |
|---|---|---|
| Mean Champion Rank | **3.5** | **3.3** |
| Median Champion Rank | 2.0 | 2.0 |
| Top-1 Accuracy | 36.8% | 36.8% |
| Top-5 Accuracy | **89.5%** | **89.5%** |
| Top-10 Accuracy | 89.5% | 89.5% |
| Top-25 Accuracy | 100% | 100% |
| Mean Brier Score | 0.044 | 0.046 |

**Year-by-year highlights (2006–2025):**

| Year | Champion | Seed | Model Rank |
|---|---|---|---|
| 2006 | Florida | 3 | **1** ✓ |
| 2008 | Kansas | 1 | **1** ✓ |
| 2012 | Kentucky | 1 | **1** ✓ |
| 2015 | Duke | 1 | 2 |
| 2018 | Villanova | 1 | **1** ✓ |
| 2022 | Kansas | 1 | **1** ✓ |
| 2023 | Connecticut | 4 | **1** ✓ |
| 2024 | Connecticut | 1 | **1** ✓ |
| 2025 | Florida | 1 | **1** ✓ (GBM) |
| 2014 | Connecticut | 7 | 13 (hardest miss) |
| 2017 | North Carolina | 1 | 11 (hardest miss) |

---

## 2026 Predictions

Based on current season statistics (as of March 2026) and Andy Katz's projected bracket:

| Rank | Seed | Team | Champion Probability |
|---|---|---|---|
| 1 | 1 | **Michigan** | 47.1% |
| 2 | 2 | **Illinois** | 30.8% |
| 3 | 1 | **Duke** | 26.1% |
| 4 | 4 | **Florida** | 25.5% |
| 5 | 1 | **Arizona** | 21.5% |
| 6 | 2 | **Houston** | 19.8% |
| 7 | 6 | **Alabama** | 14.2% |
| 8 | 4 | **Texas Tech** | 12.3% |
| 9 | 3 | **Kansas** | 10.1% |
| 10 | 1 | **UConn** | 9.7% |

*Trained on 2002–2025 (23 champions). Michigan's +38.0 adjusted efficiency margin is historically exceptional — comparable to 2025 Duke's +39.3. The model also flags Florida (defending champion, Seed 4) as a significant Cinderella-profile risk.*

---

## Features

- **Extended Dataset**: 23 years of training data (2002–2025), including reconstructed historical features for pre-2008 seasons
- **Leakage-Free Pipeline**: Imputation and scaling handled by sklearn `Pipeline` / `ColumnTransformer` — all statistics learned exclusively from training data and frozen for inference
- **Probability Normalization**: Championship probabilities are normalized to sum to 1.0 per tournament field before computing Brier Score and Log Loss — consistent with exactly one winner per year
- **Diverse Ensemble**: LogReg + GBM with skill-score + softmax dynamic weighting (see Model Details)
- **Optuna Hyperparameter Tuning**: GBM parameters are tuned per rolling-year fit via 40-trial TPE search with temporal GroupKFold CV
- **Non-Linear Seed Feature**: Replaces linear `(17 - seed)` with log of historical championship win-rate — correctly encodes the massive 1→2 seed gap
- **Era-Weighted Training**: Recent seasons are upweighted to account for concept drift (one-and-done era → transfer portal era)
- **Stable Calibration**: `CalibratedClassifierCV` now wraps a fixed-C `LogisticRegression` (C extracted from prior `LogisticRegressionCV` fit) instead of re-running nested CV on an already-tiny positive sample
- **Native Missing-Value Splits**: Entirely absent features are mapped to `NaN` (not `0`) so `HistGradientBoostingClassifier` can route on missingness natively
- **Monte Carlo Simulation**: Bracket simulation for championship odds
- **Type-Safe API**: `main.py` functions return typed `TypedDict` payloads; `results_dir` is an explicit argument (no hidden global state); unavailable years raise `ValueError` immediately

---

## Installation

```bash
git clone https://github.com/LS2026782/NCAA-Champion-Predictor.git
cd NCAA-Champion-Predictor
pip install -r requirements.txt
```

**Dependencies:** `scikit-learn`, `pandas`, `numpy`, `optuna`, `playwright` (for data collection)

---

## Usage

### Run full pipeline (backtest + 2025 predictions + Monte Carlo)
```bash
python main.py
```

### Backtest only
```bash
python main.py --backtest                  # Logistic Regression
python main.py --backtest --model gbm      # Gradient Boosting
```

### Predict a specific year
```bash
python main.py --predict 2025
python main.py --predict 2025 --model logreg
```

### 2026 bracket predictions
```bash
python predict_2026_bracket.py
```

### Monte Carlo simulation
```bash
python main.py --simulate 2025 --simulations 10000
```

---

## Project Structure

```
NCAA-Champion-Predictor/
├── config/
│   └── settings.py              # Features, model configs, Optuna config, SEED_CHAMPION_RATE
├── data/
│   ├── raw/
│   │   ├── KenPom Barttorvik Extended.csv   ← Primary training dataset (2002–2025)
│   │   ├── KenPom Barttorvik 2026.csv        ← Current season stats
│   │   └── KenPom Barttorvik.csv             ← Base dataset
│   └── historical/
│       ├── kenpom/              # Per-year KenPom CSVs (2002–2025)
│       ├── barttorvik/          # ELITE SOS data (2002–2025)
│       ├── stathead/            # Scraped EXP/class data (2003–2007)
│       └── talent/              # Recruiting data per year
├── src/
│   ├── data/
│   │   ├── loader.py            # Loads Extended CSV, creates champion labels
│   │   └── reconstruct_historical.py  # Builds Extended CSV from raw sources
│   ├── features/
│   │   ├── builder.py           # Primary feature engineering (leakage-free)
│   │   └── ultimate_builder.py  # Extended builder with WAB, FT%, momentum
│   ├── models/
│   │   ├── champion_model.py    # LogReg + GBM with Optuna tuning
│   │   └── ensemble_model.py    # LogReg+GBM ensemble with dynamic Brier weights
│   ├── evaluation/
│   │   ├── backtester.py        # Rolling-year temporal CV
│   │   └── metrics.py           # Champion rank, Brier score, Top-K rates
│   └── simulation/
│       └── monte_carlo.py       # Bracket simulation
├── results/                     # Saved backtest JSONs and prediction CSVs
├── main.py                      # Entry point
├── predict_2026_bracket.py      # 2026 championship predictions
├── run_reconstruction.py        # Rebuild the Extended CSV from raw data
└── requirements.txt
```

---

## Features Used

### Core Efficiency
| Feature | Description |
|---|---|
| `KADJ EM` | KenPom Adjusted Efficiency Margin |
| `KADJ O` | Adjusted Offensive Efficiency |
| `KADJ D` | Adjusted Defensive Efficiency |
| `BARTHAG` | Barttorvik win probability |

### Four Factors (Offense + Defense)
`EFG%`, `TOV%`, `OREB%`, `FTR` and their defensive counterparts

### Experience & Talent
| Feature | Description |
|---|---|
| `EXP` | Minutes-weighted class year (0=Fr, 3=Sr) — real data from Stathead for 2003–2007 |
| `TALENT` | Exponential-decay composite from RSCI recruiting rankings |
| `TALENT_X_EXP` | Interaction term — captures Blue Blood vs Cinderella dynamic |

### Derived Features
| Feature | Description |
|---|---|
| `SEED_STRENGTH` | `log(historical_champion_rate[seed])` — non-linear encoding |
| `EM_X_SOS` | Efficiency × SOS interaction, normalized to training-set bounds |
| `ELITE SOS` | Opponents ranked in top 50 |
| `EFG_MARGIN` | Shooting differential (offense − defense) |
| `RELATIVE_3PR` | 3-point rate relative to within-season average (concept-drift safe) |
| `CINDERELLA` | Binary: low talent + high experience + high efficiency |

---

## Leakage Prevention

The pipeline enforces strict temporal integrity at every level:

| Check | Implementation |
|---|---|
| Temporal splits | Train on `year < test_year`; never the reverse |
| Imputation | `SimpleImputer(strategy='median')` inside sklearn `Pipeline` — medians fitted on training data only, frozen for inference |
| Scaling | `StandardScaler` inside the same `Pipeline` — mean/std from training set, applied read-only to test |
| Composite features | `EM_X_SOS` and `CLUTCH_COMPOSITE` normalization bounds stored as typed instance attributes on fit, reused on transform |
| Missing columns | Entirely absent features → `NaN` (not `0`); partially missing columns → median-imputed via the pipeline |
| CV strategy | GroupKFold by season — no same-season leakage within cross-validation |
| Labels | Tournament results used only as binary `IS_CHAMPION` label, never as features |

---

## Model Details

### Logistic Regression
- L2 regularization with cross-validated `C` selection (30-point log grid)
- `class_weight='balanced'` to handle ~1:87 champion imbalance
- GroupKFold CV grouped by season (no same-year train/val contamination)

### Gradient Boosting (`HistGradientBoostingClassifier`)
- **Optuna tuning**: 40-trial TPE search per rolling-year fit
  - Tunes `learning_rate`, `max_depth`, `min_samples_leaf`, `l2_regularization`
  - Objective: Brier score via temporal GroupKFold CV
- Era weights: recent seasons upweighted to handle concept drift
- Class balancing via `sample_weight`

### Ensemble
- LogReg + GBM (diverse algorithms, not two identical models)
- **Brier Skill Score weighting**: each model's raw Brier score is first converted to a skill score relative to a naive uniform-probability baseline (`BSS = 1 − BS/BS_naive`). A model no better than guessing scores 0; negative skill is clamped to 0 so a poor model receives zero weight
- **Temperature-scaled softmax** (`T = 0.5` default) converts skill scores to weights — concentrates influence on the stronger model more aggressively than simple inverse-Brier while remaining numerically stable near zero Brier scores

---

## Data Sources

| Source | Coverage | Use |
|---|---|---|
| [KenPom](https://kenpom.com) | 2002–2025 | Efficiency metrics (KADJ EM/O/D), EXP, TALENT, SOS |
| [Barttorvik](https://barttorvik.com) | 2008–2025 | BARTHAG, ELITE SOS, Four Factors, Height |
| [Sports-Reference Stathead](https://stathead.com) | 2003–2007 | Real EXP/class data for pre-Barttorvik years |
| RSCI Recruiting Rankings | 2002–2025 | TALENT composite |

---

## Limitations

1. **Tournament variance**: Single-elimination has inherent randomness. A theoretically perfect model would still miss ~25% of champions.
2. **Cinderella floors**: Low-seeded champions (2014 UConn as a 7-seed) are near-impossible to predict from regular season stats alone.
3. **Injuries**: Pre-tournament stats don't capture roster changes between Selection Sunday and tipoff.
4. **Early era uncertainty**: 2002–2007 `EXP`, `TALENT`, and `ELITE SOS` values are partially reconstructed from approximation formulas or era averages.

---

## License

MIT — see LICENSE for details.
