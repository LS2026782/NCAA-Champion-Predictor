"""
Update 2025 NCAA tournament ROUND values in both KenPom Barttorvik.csv and
KenPom Barttorvik Extended.csv.

Champion: Florida (beat Houston 65-63 in championship game, April 7 2025)
Final Four: Florida, Houston, Auburn, Duke
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ===========================================================================
# 2025 Tournament Results — ROUND = "teams remaining at that point"
# ROUND 1 = Champion, 2 = Runner-up, 4 = Final Four, 8 = Elite 8, etc.
# ===========================================================================

ROUND_2025 = {
    # ---- CHAMPION ----
    "Florida": 1,

    # ---- RUNNER-UP ----
    "Houston": 2,

    # ---- FINAL FOUR (lost in Final Four) ----
    "Auburn": 4,
    "Duke": 4,

    # ---- ELITE EIGHT (lost in Elite Eight) ----
    "Alabama": 8,            # Lost to Duke 65-85
    "Tennessee": 8,          # Lost to Houston 50-69
    "Texas Tech": 8,         # Lost to Florida 79-84
    "Michigan St.": 8,       # Lost to Auburn 64-70

    # ---- SWEET 16 (lost in Sweet 16) ----
    "Arizona": 16,           # Lost to Duke 93-100
    "BYU": 16,               # Lost to Alabama 88-113
    "Purdue": 16,            # Lost to Houston 60-62
    "Kentucky": 16,          # Lost to Tennessee 65-78
    "Michigan": 16,          # Lost to Auburn 65-78
    "Mississippi": 16,       # Ole Miss — lost to Michigan State 70-73
    "Maryland": 16,          # Lost to Florida 71-87
    "Arkansas": 16,          # Lost to Texas Tech 83-85 (OT)

    # ---- ROUND OF 32 (lost in 2nd round) ----
    "Baylor": 32,            # Lost to Duke 66-89
    "Oregon": 32,            # Lost to Arizona 83-87
    "Wisconsin": 32,         # Lost to BYU 89-91
    "Saint Mary's": 32,      # Lost to Alabama 66-80
    "Gonzaga": 32,           # Lost to Houston 76-81
    "McNeese": 32,           # Lost to Purdue 62-76
    "Illinois": 32,          # Lost to Kentucky 75-84
    "UCLA": 32,              # Lost to Tennessee 58-67
    "Creighton": 32,         # Lost to Auburn 70-82
    "Texas A&M": 32,         # Lost to Michigan 79-91
    "Iowa St.": 32,          # Lost to Ole Miss 78-91
    "New Mexico": 32,        # Lost to Michigan State 63-71
    "Connecticut": 32,       # Lost to Florida 75-77
    "Colorado St.": 32,      # Lost to Maryland 71-72
    "Drake": 32,             # Lost to Texas Tech 64-77
    "St. John's": 32,        # Lost to Arkansas 66-75

    # ---- ROUND OF 64 (lost in 1st round) ----
    # East region
    "Mount St. Mary's": 64,  # Won First Four, lost to Duke 49-93
    "Mississippi St.": 64,   # Lost to Baylor 72-75
    "Liberty": 64,           # Lost to Oregon 52-81
    "Akron": 64,             # Lost to Arizona 65-93
    "VCU": 64,               # Lost to BYU 71-80
    "Montana": 64,           # Lost to Wisconsin 66-85
    "Vanderbilt": 64,        # Lost to Saint Mary's 56-59
    "Robert Morris": 64,     # Lost to Alabama 81-90
    # Midwest region
    "SIUE": 64,              # Lost to Houston 40-78
    "Georgia": 64,           # Lost to Gonzaga 68-89
    "Clemson": 64,           # Lost to McNeese State 67-69
    "High Point": 64,        # Lost to Purdue 63-75
    "Xavier": 64,            # Won First Four, lost to Illinois 73-86
    "Troy": 64,              # Lost to Kentucky 57-76
    "Utah St.": 64,          # Lost to UCLA 47-72
    "Wofford": 64,           # Lost to Tennessee 62-77
    # South region
    "Alabama St.": 64,       # Won First Four, lost to Auburn 63-83
    "Louisville": 64,        # Lost to Creighton 75-89
    "UC San Diego": 64,      # Lost to Michigan 65-68
    "Yale": 64,              # Lost to Texas A&M 71-80
    "North Carolina": 64,    # Won First Four, lost to Ole Miss 64-71
    "Lipscomb": 64,          # Lost to Iowa State 55-82
    "Marquette": 64,         # Lost to New Mexico 66-75
    "Bryant": 64,            # Lost to Michigan State 62-87
    # West region
    "Norfolk St.": 64,       # Lost to Florida 69-95
    "Oklahoma": 64,          # Lost to UConn 59-67
    "Memphis": 64,           # Lost to Colorado State 70-78
    "Grand Canyon": 64,      # Lost to Maryland 49-81
    "Missouri": 64,          # Lost to Drake 57-67
    "UNC Wilmington": 64,    # Lost to Texas Tech 72-82
    "Kansas": 64,            # Lost to Arkansas 72-79
    "Nebraska Omaha": 64,    # Lost to St. John's 53-83

    # ---- FIRST FOUR (play-in game losers) ----
    "American": 68,          # Lost to Mount St. Mary's 72-83
    "San Diego St.": 68,     # Lost to UNC 68-95
    "Texas": 68,             # Lost to Xavier 80-86
    "Saint Francis": 68,     # Lost to Alabama State 68-70
}


def update_rounds(filepath: Path, rounds: dict, year: int = 2025):
    """Update ROUND values for tournament teams in the given year."""
    df = pd.read_csv(filepath)

    mask_year = df["YEAR"] == year
    tourney_mask = df["SEED"].notna() & mask_year

    # Count current state
    before_r1 = (df["ROUND"] == 1).sum()

    updated = 0
    not_found = []
    for team, rnd in rounds.items():
        match = (df["TEAM"] == team) & mask_year
        if match.any():
            df.loc[match, "ROUND"] = float(rnd)
            updated += 1
        else:
            not_found.append(team)

    after_r1 = (df["ROUND"] == 1).sum()

    df.to_csv(filepath, index=False)

    print(f"\n{filepath.name}:")
    print(f"  Updated {updated}/{len(rounds)} teams")
    print(f"  Champions before: {before_r1}, after: {after_r1}")
    if not_found:
        print(f"  NOT FOUND in dataset ({len(not_found)}): {not_found}")

    # Verify
    d25 = df[df["YEAR"] == year]
    print(f"\n  2025 ROUND distribution:")
    print(f"  {d25['ROUND'].value_counts().sort_index().to_dict()}")
    champ = d25[d25["ROUND"] == 1]["TEAM"].values
    runner = d25[d25["ROUND"] == 2]["TEAM"].values
    ff = d25[d25["ROUND"] == 4]["TEAM"].values
    print(f"  Champion: {list(champ)}")
    print(f"  Runner-up: {list(runner)}")
    print(f"  Final Four: {list(ff)}")

    return df


if __name__ == "__main__":
    print("Updating 2025 tournament ROUND values...")
    print("Champion: Florida | Runner-up: Houston")
    print("Final Four: Auburn, Duke, Florida, Houston")

    # Update base CSV
    update_rounds(RAW_DIR / "KenPom Barttorvik.csv", ROUND_2025)

    # Update Extended CSV
    update_rounds(RAW_DIR / "KenPom Barttorvik Extended.csv", ROUND_2025)

    print("\nDone! Running validation...")
    import subprocess
    result = subprocess.run(
        ["python", "run_reconstruction.py", "--validate"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True
    )
    # Show champion/year summary from validation
    lines = result.stdout.split("\n")
    for line in lines:
        if "Champion" in line or "2024" in line or "2025" in line or "Year" in line:
            print(line)
