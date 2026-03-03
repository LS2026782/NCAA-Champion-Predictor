"""
Final integration script: merge all collected data into main dataset.

Column semantics (from KenPom):
  AVG HGT   = minutes-weighted average height (inches, e.g. 76.5)
  EFF HGT   = deviation from D1 average by position (inches, range ~ -7 to +9)
              0.0 = exactly league average; fill missing with 0.0
  EXP       = minutes-weighted experience (0=all-frosh, 3=all-seniors)
  ELITE SOS = Barttorvik Elite SOS (% of top-50 opponents faced, or approximated)
  TALENT    = placeholder column (NaN); filled by reconstruct_historical.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HIST_DIR = PROJECT_ROOT / "data" / "historical"
BART_DIR = HIST_DIR / "barttorvik"
ROSTER_DIR = HIST_DIR / "rosters"


def fuzzy_match(name, candidates):
    """Return best candidate match or None."""
    name_l = name.lower().replace(".", "").replace("'", "")
    words = set(name_l.split())
    best, best_score = None, 0
    for c in candidates:
        c_l = c.lower().replace(".", "").replace("'", "")
        c_words = set(c_l.split())
        inter = words & c_words
        union = words | c_words
        score = len(inter) / max(len(union), 1)
        if score > best_score and score > 0.5:
            best, best_score = c, score
    return best


def main():
    df = pd.read_csv(RAW_DIR / "KenPom Barttorvik.csv")
    print(f"Main dataset: {len(df)} rows, {len(df.columns)} columns")

    # Save clean backup
    df.to_csv(RAW_DIR / "KenPom Barttorvik_backup.csv", index=False)

    kp_teams = set(df["TEAM"].unique())

    # ── 1. ELITE SOS ─────────────────────────────────────────────────────────
    esos_file = BART_DIR / "elite_sos_all.csv"
    if esos_file.exists():
        esos = pd.read_csv(esos_file)
        print(f"\nElite SOS records: {len(esos)}")

        if "ELITE SOS" not in df.columns:
            df["ELITE SOS"] = np.nan

        bart_teams = set(esos["TEAM"].unique())
        name_map = {}
        unmatched = kp_teams - bart_teams
        for kp_name in unmatched:
            m = fuzzy_match(kp_name, bart_teams)
            if m:
                name_map[m] = kp_name

        # Apply with name normalisation
        filled = 0
        for _, row in esos.iterrows():
            kp_name = name_map.get(row["TEAM"], row["TEAM"])
            mask = (df["TEAM"] == kp_name) & (df["YEAR"] == row["YEAR"])
            if mask.any() and pd.isna(df.loc[mask, "ELITE SOS"]).all():
                df.loc[mask, "ELITE SOS"] = row["ELITE SOS"]
                filled += mask.sum()

        valid = df["ELITE SOS"].notna().sum()
        print(f"  Filled: {filled} | Total valid: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")
    else:
        print("\nNo ELITE SOS file found — run --phase1 first")

    # ── 2. EXP & AVG HGT from Sports-Reference (2002-2006) ──────────────────
    roster_files = list(ROSTER_DIR.glob("roster_200*.csv"))
    if roster_files:
        roster_dfs = [pd.read_csv(f) for f in roster_files]
        roster_all = pd.concat(roster_dfs, ignore_index=True)
        print(f"\nRoster records: {len(roster_all)}, EXP valid: {roster_all['EXP'].notna().sum()}")

        filled_exp = 0
        for _, row in roster_all.iterrows():
            if pd.isna(row.get("EXP")):
                continue
            mask = (df["TEAM"] == row["TEAM"]) & (df["YEAR"] == row["YEAR"])
            if mask.any() and pd.isna(df.loc[mask, "EXP"]).all():
                df.loc[mask, "EXP"] = row["EXP"]
                # AVG HGT from Sports-Reference is real inches — store directly
                if pd.notna(row.get("AVG HGT")):
                    df.loc[mask, "AVG HGT"] = row["AVG HGT"]
                # EFF HGT from Sports-Reference is also absolute inches, but the
                # KenPom column is a deviation. Leave EFF HGT as 0.0 (neutral)
                # for years without KenPom data.
                filled_exp += 1

        print(f"  Filled EXP/AVG HGT for {filled_exp} team-years")
    else:
        print("\nNo roster files found — run fetch_sr_rosters.py first")

    # ── 3. Fill remaining 2002-2006 missing values ───────────────────────────
    # EXP: use era-appropriate average (veteran era 2007-2010)
    era_data = df[(df["YEAR"] >= 2007) & (df["YEAR"] <= 2010) & df["EXP"].notna()]
    era_exp = era_data["EXP"].mean() if len(era_data) > 0 else 1.65

    era_hgt_data = df[(df["YEAR"] >= 2007) & (df["YEAR"] <= 2010) & (df["AVG HGT"] > 70)]
    era_hgt = era_hgt_data["AVG HGT"].mean() if len(era_hgt_data) > 0 else 76.5

    still_missing_exp = df["EXP"].isna() & df["YEAR"].isin([2002, 2003, 2004, 2005, 2006])
    still_missing_hgt = (df["AVG HGT"].isna() | (df["AVG HGT"] < 70)) & df["YEAR"].isin(
        [2002, 2003, 2004, 2005, 2006]
    )

    print(f"\nEra averages (2007-2010): EXP={era_exp:.3f}, AVG HGT={era_hgt:.1f}")
    print(f"Still missing EXP: {still_missing_exp.sum()} rows — filling with era avg")
    print(f"Still missing/invalid AVG HGT: {still_missing_hgt.sum()} rows — filling with era avg")

    df.loc[still_missing_exp, "EXP"] = era_exp
    df.loc[still_missing_hgt, "AVG HGT"] = era_hgt

    # EFF HGT: deviation from average — 0.0 = perfectly average
    # For 2002-2006 rows without KenPom position data, 0.0 is the correct imputation
    still_missing_eff = df["EFF HGT"].isna()
    df.loc[still_missing_eff, "EFF HGT"] = 0.0
    print(f"EFF HGT: {still_missing_eff.sum()} rows filled with 0.0 (neutral)")

    # ── 4. TALENT placeholder ────────────────────────────────────────────────
    if "TALENT" not in df.columns:
        df["TALENT"] = np.nan
    # Will be filled by run_reconstruction.py with era-appropriate values

    # ── 5. Save ──────────────────────────────────────────────────────────────
    df.to_csv(RAW_DIR / "KenPom Barttorvik.csv", index=False)

    print("\n" + "=" * 55)
    print("FINAL COVERAGE REPORT")
    print("=" * 55)
    key_cols = ["ELITE SOS", "TALENT", "EXP", "AVG HGT", "EFF HGT"]
    for col in key_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            pct = valid / len(df) * 100
            status = "COMPLETE" if pct >= 99 else f"{pct:.1f}%"
            print(f"  {col:12}: {valid:5}/{len(df)} ({status})")
            if valid < len(df):
                miss_yrs = sorted(df[df[col].isna()]["YEAR"].unique())
                if miss_yrs:
                    print(f"               Missing years: {miss_yrs}")

    print(f"\nSaved: {len(df)} rows, {len(df.columns)} columns")
    print("\nNext: python run_reconstruction.py --run  (fills TALENT with era averages)")
    print("      python fetch_sr_rosters.py           (rescrape after IP cooldown)")


if __name__ == "__main__":
    main()
