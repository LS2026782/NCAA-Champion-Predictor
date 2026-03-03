"""Final validation of the integrated dataset."""
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/KenPom Barttorvik Extended.csv")
print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Years: {sorted(df['YEAR'].unique())[0]} - {sorted(df['YEAR'].unique())[-1]}")

print("\n=== FEATURE COVERAGE ===")
for col in ["ELITE SOS", "TALENT", "EXP", "AVG HGT", "EFF HGT",
            "BARTHAG", "KADJ O", "KADJ D", "KADJ EM"]:
    if col in df.columns:
        valid = df[col].notna().sum()
        pct = valid / len(df) * 100
        status = "COMPLETE" if pct >= 99 else f"{pct:.1f}%"
        print(f"  {col:12}: {valid:5}/{len(df)} ({status})")

print("\n=== TOURNAMENT TEAMS: EARLY YEARS ===")
for yr in [2002, 2003, 2004]:
    tourney = df[(df["YEAR"] == yr) & df["ROUND"].notna()]
    print(f"\n  {yr} top-5 by KADJ EM:")
    top = tourney.nlargest(5, "KADJ EM")
    for _, r in top.iterrows():
        print(f"    {r['TEAM']:25} | EM={r['KADJ EM']:+.1f} | "
              f"EXP={r['EXP']:.2f} | HGT={r['AVG HGT']:.1f} | "
              f"ESOS={r.get('ELITE SOS', float('nan')):.1f}")

print("\n=== EXP BY ERA ===")
for era_name, yr_range in [
    ("Veteran (2002-2009)", range(2002, 2010)),
    ("One-and-done (2010-2019)", range(2010, 2020)),
    ("Transfer portal (2021-2025)", range(2021, 2026)),
]:
    era = df[df["YEAR"].isin(yr_range)]
    print(f"  {era_name}: EXP={era['EXP'].mean():.2f}, AVG HGT={era['AVG HGT'].mean():.1f}")

print("\n=== TALENT DISTRIBUTION ===")
talent = df[df["TALENT"].notna()]
print(f"  Total with TALENT: {len(talent)}")
if len(talent) > 0:
    print(f"  Range: {talent['TALENT'].min():.1f} - {talent['TALENT'].max():.1f}")
    print(f"  Mean : {talent['TALENT'].mean():.1f}")

print("\n=== CHAMPIONS CHECK ===")
champs = df[df["ROUND"] == 1]
print(f"  Total champions: {len(champs)}")
for _, r in champs.sort_values("YEAR").iterrows():
    print(f"  {r['YEAR']} {r['TEAM']:25} EXP={r['EXP']:.2f} TALENT={r['TALENT']:.1f} "
          f"ESOS={r.get('ELITE SOS', float('nan')):.1f}")
