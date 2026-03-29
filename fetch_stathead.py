"""
Fetch EXP and HGT data for 2002-2006 using Stathead College Basketball
Player Season Finder.

Stathead allows bulk CSV export of all players across all teams in a
season range — far faster than scraping individual team pages.

Strategy:
  1. Open a visible browser so you can log in to Stathead
  2. Navigate to College Basketball Player Season Finder
  3. Export all 2002-2006 players with class, height, minutes per game
  4. Process into team-level EXP and AVG HGT
  5. Merge into main dataset

Run with:
    python fetch_stathead.py
"""

import asyncio
import re
import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STATHEAD_DIR = DATA_DIR / "historical" / "stathead"
STATHEAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_MAP = {
    "fr": 0, "freshman": 0,
    "so": 1, "sophomore": 1,
    "jr": 2, "junior": 2,
    "sr": 3, "senior": 3,
    "gr": 3, "graduate": 3,
    "rs fr": 0, "rs so": 1, "rs jr": 2, "rs sr": 3,
}


def parse_height_inches(h):
    """Convert '6-8' or '6\'8"' to inches (80)."""
    if pd.isna(h) or not str(h).strip():
        return None
    h = str(h).strip()
    m = re.match(r"(\d+)-(\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    m = re.match(r"(\d+)['\u2019](\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    try:
        v = float(h)
        return int(v) if v > 12 else int(v * 12)
    except ValueError:
        return None


def compute_team_metrics(players_df):
    """
    Compute EXP and AVG HGT from a DataFrame of players.

    Expects columns: class_val, height_in, minutes (total season minutes)
    """
    df = players_df[players_df["minutes"] > 0].copy()
    if df.empty:
        df = players_df.copy()
        df["minutes"] = 100.0

    total = df["minutes"].sum()
    if total == 0:
        return None

    exp = (df["class_val"] * df["minutes"]).sum() / total

    hgt = df[df["height_in"].notna()].copy()
    avg_hgt = None
    if not hgt.empty:
        hgt_mins = hgt["minutes"].sum()
        if hgt_mins > 0:
            avg_hgt = (hgt["height_in"] * hgt["minutes"]).sum() / hgt_mins

    return {
        "EXP": round(exp, 3),
        "AVG HGT": round(avg_hgt, 1) if avg_hgt else None,
        "n_players": len(df),
    }


def process_stathead_csv(csv_path):
    """
    Process a Stathead Player Season Finder CSV export into
    team-level EXP and AVG HGT metrics.

    Stathead column names vary slightly — we normalize them.
    """
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} rows from {csv_path.name}")
    print(f"Columns: {list(df.columns)}")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]", "_", regex=True)
    print(f"Normalized columns: {list(df.columns)[:20]}")

    # Column mapping: Stathead → standard names
    col_map = {
        # School/team
        "school": "TEAM", "team": "TEAM", "school_name": "TEAM",
        # Year
        "year": "YEAR", "season": "YEAR",
        # Class
        "class": "CLASS", "class_": "CLASS", "yr": "CLASS",
        # Height
        "height": "HEIGHT", "ht": "HEIGHT", "ht_": "HEIGHT",
        # Minutes
        "mp": "MP_PER_G", "mp_per_g": "MP_PER_G", "min": "MP_PER_G",
        "mp_per_game": "MP_PER_G",
        # Games
        "g": "GAMES", "games": "GAMES", "g_": "GAMES",
    }

    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    print(f"\nAfter mapping, available standard cols: "
          f"{[c for c in ['TEAM','YEAR','CLASS','HEIGHT','MP_PER_G','GAMES'] if c in df.columns]}")

    # Parse class to numeric
    if "CLASS" in df.columns:
        df["class_val"] = df["CLASS"].astype(str).str.strip().str.lower().map(
            lambda x: CLASS_MAP.get(x, 1.5)
        )
    else:
        print("WARNING: CLASS column not found — using 1.5 for all players")
        df["class_val"] = 1.5

    # Parse height
    if "HEIGHT" in df.columns:
        df["height_in"] = df["HEIGHT"].apply(parse_height_inches)
    else:
        print("WARNING: HEIGHT column not found")
        df["height_in"] = None

    # Compute total minutes
    if "MP_PER_G" in df.columns and "GAMES" in df.columns:
        df["minutes"] = pd.to_numeric(df["MP_PER_G"], errors="coerce").fillna(0) * \
                        pd.to_numeric(df["GAMES"], errors="coerce").fillna(0)
    elif "MP_PER_G" in df.columns:
        df["minutes"] = pd.to_numeric(df["MP_PER_G"], errors="coerce").fillna(0) * 30
    else:
        print("WARNING: No minutes column found — using equal weights")
        df["minutes"] = 100.0

    # Normalise team name and year
    if "TEAM" not in df.columns:
        # Look for any column that might be the school
        for col in df.columns:
            if "school" in col or "team" in col:
                df = df.rename(columns={col: "TEAM"})
                break

    if "YEAR" not in df.columns:
        for col in df.columns:
            if "year" in col or "season" in col:
                df = df.rename(columns={col: "YEAR"})
                break

    if "TEAM" not in df.columns or "YEAR" not in df.columns:
        print("ERROR: Could not identify TEAM or YEAR columns. Exiting.")
        print("Available columns:", list(df.columns))
        return pd.DataFrame()

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df.dropna(subset=["TEAM", "YEAR"])

    # Compute team metrics
    results = []
    for (team, year), group in df.groupby(["TEAM", "YEAR"]):
        metrics = compute_team_metrics(group)
        if metrics:
            results.append({"TEAM": team, "YEAR": int(year), **metrics})

    out_df = pd.DataFrame(results)
    print(f"\nComputed metrics for {len(out_df)} team-years")
    print(f"Years covered: {sorted(out_df['YEAR'].unique()) if not out_df.empty else []}")
    return out_df


def integrate_into_main(metrics_df):
    """Merge Stathead-derived EXP/HGT into the main dataset."""
    main_path = RAW_DIR / "KenPom Barttorvik.csv"
    df = pd.read_csv(main_path)

    kp_teams = set(df["TEAM"].unique())
    sh_teams = set(metrics_df["TEAM"].unique())

    # Build fuzzy name map for mismatches
    def fuzzy_match(name, candidates):
        name_l = name.lower().replace(".", "").replace("'", "")
        words = set(name_l.split())
        best, best_score = None, 0
        for c in candidates:
            c_l = c.lower().replace(".", "").replace("'", "")
            c_words = set(c_l.split())
            inter = words & c_words
            union = words | c_words
            score = len(inter) / max(len(union), 1)
            if score > best_score and score > 0.55:
                best, best_score = c, score
        return best

    only_sh = sh_teams - kp_teams
    name_map = {}
    for sh_name in only_sh:
        m = fuzzy_match(sh_name, kp_teams)
        if m:
            name_map[sh_name] = m

    filled_exp = 0
    for _, row in metrics_df.iterrows():
        kp_name = name_map.get(row["TEAM"], row["TEAM"])
        mask = (df["TEAM"] == kp_name) & (df["YEAR"] == row["YEAR"])
        if not mask.any():
            continue

        if pd.notna(row.get("EXP")) and pd.isna(df.loc[mask, "EXP"]).all():
            df.loc[mask, "EXP"] = row["EXP"]
            filled_exp += 1

        if pd.notna(row.get("AVG HGT")) and (
            pd.isna(df.loc[mask, "AVG HGT"]).all()
            or (df.loc[mask, "AVG HGT"] < 70).all()
        ):
            df.loc[mask, "AVG HGT"] = row["AVG HGT"]

    print(f"\nFilled EXP for {filled_exp} team-years")

    # Report remaining gaps
    still_missing = df["EXP"].isna() & df["YEAR"].isin([2002, 2003, 2004, 2005, 2006])
    if still_missing.any():
        miss_teams = df.loc[still_missing, ["TEAM", "YEAR"]].drop_duplicates()
        print(f"{still_missing.sum()} team-years still missing EXP")
        print("  (These will be filled with era averages by run_reconstruction.py)")

    df.to_csv(main_path, index=False)
    print(f"Saved updated dataset: {len(df)} rows")

    print("\nCoverage after Stathead integration:")
    for col in ["EXP", "AVG HGT", "ELITE SOS"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            pct = valid / len(df) * 100
            print(f"  {col:12}: {valid}/{len(df)} ({pct:.1f}%)")

    return df


async def open_stathead_and_wait():
    """
    Open a visible Chromium browser pointing at Stathead's
    College Basketball Player Season Finder.

    The user logs in manually, sets filters, and downloads the CSV.
    The script then processes the downloaded files automatically.
    """
    from playwright.async_api import async_playwright

    print("=" * 62)
    print("STATHEAD DATA COLLECTION")
    print("=" * 62)
    print()
    print("Opening Stathead in a visible browser window.")
    print("Follow the steps below to download the data:\n")

    steps = """
STEP-BY-STEP INSTRUCTIONS
--------------------------
1. Log in to your Stathead account when the browser opens.

2. Navigate to:
   Basketball -> College -> Player Season Finder

   Or go directly to:
   https://www.sports-reference.com/stathead/college-basketball/player-season-finder/

3. Set these filters:
   - Season type: Regular Season + Postseason (All)
   - Seasons: FROM 2002  TO 2006
   - Leave all other filters as default

4. Click "Find Players"

5. When results load, scroll to the top and click "CSV" to download.
   Save the file to:
   C:\\Users\\nicho\\NCAA-Champion-Predictor\\data\\historical\\stathead\\

   Name the file: players_2002_2006.csv

6. If the results are paginated (200 rows max), repeat with offsets:
   - Page 1: offset=0   -> save as players_2002_2006_p1.csv
   - Page 2: offset=200 -> save as players_2002_2006_p2.csv
   - ... continue until all pages downloaded

7. Press ENTER in this terminal when all files are saved.
"""

    print(steps)

    FINDER_URL = (
        "https://www.sports-reference.com/stathead/college-basketball/"
        "player-season-finder/"
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
        )
        page = await ctx.new_page()

        # Go to Stathead login first
        print("Opening Stathead...")
        await page.goto("https://www.sports-reference.com/account/login/",
                        wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)

        print(f"\nNavigating to College Basketball Player Season Finder...")
        await page.goto(FINDER_URL, wait_until="domcontentloaded", timeout=30000)

        print("\nBrowser is open. Follow the steps above.")
        print(f"Watching for CSV files in: {STATHEAD_DIR}")
        print("Script will auto-continue when files appear. (Ctrl+C to abort)\n")

        # Poll for downloaded files instead of waiting for stdin
        seen_files = set(STATHEAD_DIR.glob("players_*.csv"))
        stable_count = 0

        while True:
            await asyncio.sleep(5)
            current_files = set(STATHEAD_DIR.glob("players_*.csv"))
            new_files = current_files - seen_files

            if new_files:
                for f in new_files:
                    print(f"  Detected: {f.name}")
                seen_files = current_files
                stable_count = 0
            elif seen_files:
                stable_count += 1
                if stable_count >= 4:  # 4 x 5s = 20s stable = all downloads done
                    print(f"\nAll {len(seen_files)} file(s) stable. Proceeding...")
                    break
                print(f"  {len(seen_files)} file(s) ready, waiting for stability ({stable_count}/4)...")

        await browser.close()

    return True


def process_all_stathead_files():
    """Process all downloaded Stathead CSV files and merge into main dataset."""
    csv_files = sorted(STATHEAD_DIR.glob("players_*.csv"))

    if not csv_files:
        print(f"\nNo CSV files found in {STATHEAD_DIR}")
        print("Please download from Stathead first (run without --process-only)")
        return

    print(f"\nFound {len(csv_files)} CSV files to process:")
    for f in csv_files:
        print(f"  {f.name}")

    all_metrics = []
    for csv_path in csv_files:
        metrics = process_stathead_csv(csv_path)
        if not metrics.empty:
            all_metrics.append(metrics)

    if not all_metrics:
        print("\nNo valid data extracted from CSV files.")
        return

    combined = pd.concat(all_metrics, ignore_index=True)
    combined = combined.drop_duplicates(subset=["TEAM", "YEAR"], keep="last")

    print(f"\nCombined metrics: {len(combined)} team-years")
    print(f"Years: {sorted(combined['YEAR'].unique())}")

    # Save combined metrics
    combined.to_csv(STATHEAD_DIR / "team_metrics_2002_2006.csv", index=False)

    # Merge into main dataset
    df = integrate_into_main(combined)

    # Now run the reconstruction pipeline to fill any remaining gaps
    print("\nRunning reconstruction pipeline to fill remaining gaps...")
    import subprocess
    result = subprocess.run(
        ["python", "run_reconstruction.py", "--run"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    if result.returncode == 0:
        print("Reconstruction complete.")
    else:
        print("Reconstruction error:", result.stderr[:500])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-only", action="store_true",
                        help="Process already-downloaded CSV files without opening browser")
    parser.add_argument("--show-columns", type=str, default=None,
                        help="Show columns of a specific CSV file")
    args = parser.parse_args()

    if args.show_columns:
        df = pd.read_csv(args.show_columns)
        print(f"File: {args.show_columns}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())
        return

    if args.process_only:
        process_all_stathead_files()
    else:
        asyncio.run(open_stathead_and_wait())
        print("\nNow processing downloaded files...")
        process_all_stathead_files()


if __name__ == "__main__":
    main()
