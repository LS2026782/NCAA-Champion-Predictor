# Historical Data Reconstruction Pipeline

This document describes the system for extending the NCAA Championship Predictor training data from 2008-2025 back to 2002-2025.

## Overview

Based on the research document "LONGITUDINAL ARCHITECTURE FOR PREDICTIVE MODELING IN COLLEGIATE BASKETBALL", this pipeline implements **Option B (Feature Reconstruction)** - creating a unified, homogeneous feature space across all 24 years.

### Why This Matters

- Current training data: **17 seasons** (2008-2025), ~17 champion examples
- Extended training data: **23 seasons** (2002-2025, excluding 2020), ~23 champion examples
- **35% more training data** for the rare-event classification problem

### Key Insight from Research

> "In rare event prediction, every positive sample is precious. Discarding the champions of 2002-2013 deprives the model of 50% of the available 'truth' data."

## Files Created

### Core Pipeline Scripts

| File | Purpose |
|------|---------|
| `run_reconstruction.py` | **Master script** - Status check, guide, run pipeline, validate |
| `src/data/reconstruct_historical.py` | Core reconstruction functions (BARTHAG, TALENT, EXP, HEIGHT) |
| `src/data/fetch_kenpom_historical.py` | KenPom data acquisition guide and parser |
| `src/data/fetch_rsci_data.py` | RSCI recruiting data fetcher |

### Key Formulas Implemented

**BARTHAG** (Win Probability):
```
BARTHAG = AdjO^11.5 / (AdjO^11.5 + AdjD^11.5)
```

**TALENT** (Recruiting Quality):
```
Player Score = 100 * e^(-(RSCI_Rank - 1) / 25)
Team TALENT = SUM(Player_Score * Minutes) / SUM(Minutes)
```

**EXPERIENCE** (Roster Maturity):
```
Class Values: Fr=0, So=1, Jr=2, Sr=3
Team EXP = SUM(Class_Value * Minutes) / SUM(Minutes)
```

**EFFECTIVE HEIGHT** (Interior Size):
```
Average height of players comprising top 40% of minutes (sorted by height)
```

## Quick Start

### Step 1: Check Current Status
```bash
python run_reconstruction.py --status
```

### Step 2: View Full Guide
```bash
python run_reconstruction.py --guide
```

### Step 3: Acquire KenPom Data
1. Subscribe to kenpom.com ($25/year)
2. Download efficiency tables for 2002-2007
3. Save to `data/historical/kenpom/kenpom_YYYY.csv`

### Step 4: Run Reconstruction
```bash
python run_reconstruction.py --run
```

### Step 5: Validate Results
```bash
python run_reconstruction.py --validate
```

## Data Requirements

### Must Have (KenPom Subscription)
- AdjO, AdjD, AdjEM, AdjT for 2002-2007
- Four Factors (eFG%, TO%, ORB%, FTR - offense and defense)
- Can calculate BARTHAG automatically

### Nice to Have (For Full Reconstruction)
- RSCI recruiting rankings (1998-2007 classes)
- Sports-Reference roster data (class, height, minutes)
- Tournament seeds and results for 2002-2007

### Fallback Options
If full roster data unavailable:
- TALENT: Approximate by conference (Power 5 = ~50, Mid-major = ~25)
- EXP: Use default value of 1.5
- HEIGHT: Use default values (AVG HGT = 76", EFF HGT = 80")

## Directory Structure

```
data/
├── raw/
│   ├── KenPom Barttorvik.csv          # Existing 2008-2025 data
│   └── KenPom Barttorvik Extended.csv # Output: 2002-2025 data
└── historical/
    ├── kenpom/                         # Historical KenPom CSVs
    │   ├── kenpom_2002.csv
    │   ├── kenpom_2003.csv
    │   └── ...
    ├── rsci/                           # RSCI recruiting rankings
    │   ├── rsci_1998.csv
    │   ├── rsci_1999.csv
    │   └── ...
    └── rosters/                        # Sports-Reference roster data
        ├── roster_duke_2003.csv
        └── ...
```

## Updating settings.py

After reconstruction, update `config/settings.py`:

```python
# Change from:
MIN_YEAR = 2008

# To:
MIN_YEAR = 2002
```

## Validation Checklist

After reconstruction, verify:
- [ ] 23 seasons in dataset (2002-2025, no 2020)
- [ ] All required columns present
- [ ] BARTHAG values between 0-1
- [ ] KADJ EM values between -30 and +40
- [ ] Tournament seeds/rounds for 2002-2007 tourney teams
- [ ] No unexpected missing values in core metrics

## Next Steps After Data Acquisition

1. **Get KenPom subscription** - Essential first step
2. **Download 2002-2007 data** - Follow the guide
3. **Add tournament data** - Seeds and results for historical tournaments
4. **Optional: RSCI data** - For accurate TALENT reconstruction
5. **Run pipeline** - Generate extended dataset
6. **Retrain models** - With 35% more champion examples
