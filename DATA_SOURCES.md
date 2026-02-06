# NCAA Champion Predictor - Data Sources

## Current Season Data (2025-26)

### Primary Source: Barttorvik (FREE)

**URL:** https://barttorvik.com/trank.php?year=2026

Barttorvik provides all the advanced metrics we need for free:
- Adjusted Efficiency (AdjEM, AdjO, AdjD)
- BARTHAG (win probability)
- TALENT (recruiting composite) - **our most predictive new feature!**
- Experience, Height, Four Factors, etc.

**How to download:**
1. Visit the URL above
2. The main table shows all D1 teams with their metrics
3. Look for CSV/export option, or use the customizable tables:
   - https://barttorvik.com/team-tables_each.php
4. Save as: `data/raw/barttorvik_2026.csv`

### Secondary Source: KenPom ($24.95/year)

**URL:** https://kenpom.com/

KenPom provides similar metrics with slightly different calculations:
- KADJ EM, KADJ O, KADJ D (adjusted efficiency)
- Tempo data
- Historical data back to 1999

**Automated access via Python:**
```python
pip install kenpompy

from kenpompy.utils import login
import kenpompy.summary as kp

browser = login('your_email', 'your_password')
ratings = kp.get_efficiency(browser, season='2026')
```

## Data Pipeline

### Step 1: Download Current Season Data
```bash
# View instructions
python -m src.data.fetch_current_season --source instructions

# If you have KenPom credentials
python -m src.data.fetch_current_season --source kenpom --year 2026 --email EMAIL --password PASS
```

### Step 2: Merge Into Unified Format
```bash
# Preview the merge
python -m src.data.merge_season_data --year 2026 --preview

# Save as separate file
python -m src.data.merge_season_data --year 2026

# Append to main dataset
python -m src.data.merge_season_data --year 2026 --append
```

### Step 3: Add Tournament Seeds (After Selection Sunday)
After the bracket is announced, update the SEED column in the data file for tournament teams.

### Step 4: Run Predictions
```bash
# Champion predictions
python main.py --year 2026

# Game-by-game predictions
python predict_games.py --year 2026
```

## Required Columns

Our models need these columns (importance ranked):

| Column | Description | Source | Importance |
|--------|-------------|--------|------------|
| KADJ EM | Adjusted Efficiency Margin | KenPom/Barttorvik | Critical |
| BARTHAG | Win probability metric | Barttorvik | Critical |
| TALENT | Recruiting composite | Barttorvik | High (+1.1% accuracy) |
| SEED | Tournament seed | Manual | High |
| EFG% | Effective FG% | Both | Medium |
| TOV% | Turnover rate | Both | Medium |
| OREB% | Offensive rebound rate | Both | Medium |
| ELITE SOS | Strength of schedule | Barttorvik | Medium |
| EXP | Team experience | Both | Low |
| AVG HGT | Average height | Both | Low |

## Updating for Future Seasons

1. Download fresh data from Barttorvik (free) or KenPom
2. Run the merge script to standardize column names
3. Add tournament seeds after Selection Sunday
4. Run predictions!

## Links

- **Barttorvik:** https://barttorvik.com
- **KenPom:** https://kenpom.com
- **kenpompy docs:** https://kenpompy.readthedocs.io
- **CBBpy (ESPN data):** https://github.com/dcstats/CBBpy
