"""Fix EFF HGT zero/null values in the main dataset."""
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/KenPom Barttorvik.csv")

print("EFF HGT describe:")
print(df["EFF HGT"].describe())
print(f"\nNon-zero EFF HGT years: {sorted(df[df['EFF HGT'] > 0]['YEAR'].unique())}")

# Where EFF HGT is 0 or null but AVG HGT is valid, use AVG HGT + 2.5in
bad_eff = (df["EFF HGT"].isna()) | (df["EFF HGT"] == 0)
has_avg = df["AVG HGT"].notna() & (df["AVG HGT"] > 0)

combo = bad_eff & has_avg
df.loc[combo, "EFF HGT"] = df.loc[combo, "AVG HGT"] + 2.5
print(f"\nFixed {combo.sum()} rows using AVG HGT + 2.5")

# Any remaining nulls — use global mean
still_bad = df["EFF HGT"].isna() | (df["EFF HGT"] == 0)
global_mean = df.loc[~still_bad, "EFF HGT"].mean()
df.loc[still_bad, "EFF HGT"] = global_mean
print(f"Filled {still_bad.sum()} remaining with global mean {global_mean:.1f}")

df.to_csv("data/raw/KenPom Barttorvik.csv", index=False)

print("\nFinal EFF HGT stats:")
print(f"  range : {df['EFF HGT'].min():.1f} - {df['EFF HGT'].max():.1f}")
print(f"  mean  : {df['EFF HGT'].mean():.1f}")
print(f"  null  : {df['EFF HGT'].isna().sum()}")

print("\nFinal dataset coverage:")
for col in ["ELITE SOS", "TALENT", "EXP", "AVG HGT", "EFF HGT"]:
    if col in df.columns:
        valid = df[col].notna().sum()
        print(f"  {col:12}: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")
