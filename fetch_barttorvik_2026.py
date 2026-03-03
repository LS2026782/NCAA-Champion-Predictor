"""
Fetch and process 2026 Barttorvik data.
Uses Playwright to:
  1) Download the raw CSV (clean team names + stats)
  2) Scrape DOM for conference info
  3) Scrape SOS page for Elite SOS
Then merges and outputs pipeline-ready CSV.
"""
import asyncio
import re
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "barttorvik_2026_raw.csv"
OUTPUT = RAW_DIR / "barttorvik_2026.csv"

CSV_COLUMNS = {
    0: "Team", 1: "AdjOE", 2: "AdjDE", 3: "Barthag",
    4: "Rec", 5: "W", 6: "G",
    7: "EFG%", 8: "EFGD%", 9: "FTR", 10: "FTRD",
    11: "TOR", 12: "TORD", 13: "ORB", 14: "DRB",
    15: "AdjT", 16: "2P%", 17: "2P%D", 18: "3P%", 19: "3P%D",
    20: "_seed1", 21: "_seed2",
    22: "_unknown1", 23: "_unknown2",
    24: "3PR", 25: "3PRD",
    26: "_adjt_dup",
    30: "YEAR",
    34: "WAB", 35: "Hgt",
}

def clean_team_name(raw):
    """Strip upcoming-game annotations."""
    m = re.match(r"^(.+)\s+\([AHN]\)\s+\d+\s", raw)
    return m.group(1).strip() if m else raw.strip()


async def scrape_dom_table(page):
    """Extract team→conference mapping from DOM."""
    data = await page.evaluate(r"""() => {
        const tables = document.querySelectorAll('table');
        let best = null, mx = 0;
        tables.forEach(t => {
            const n = t.querySelectorAll('tbody tr').length;
            if (n > mx) { mx = n; best = t; }
        });
        if (!best) return null;
        const hdrs = [];
        best.querySelectorAll('thead th, thead td').forEach(c =>
            hdrs.push(c.innerText.trim().replace(/\n/g, ' '))
        );
        const rows = [];
        best.querySelectorAll('tbody tr').forEach(r => {
            const cells = [];
            r.querySelectorAll('td').forEach(c =>
                cells.push(c.innerText.trim().replace(/\n/g, ' '))
            );
            if (cells.length > 5) rows.push(cells);
        });
        return { headers: hdrs, rows: rows };
    }""")
    return data


async def fetch_all():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            accept_downloads=True,
        )
        page = await ctx.new_page()

        print("=" * 70)
        print("FETCHING 2026 BARTTORVIK DATA (as of 2026-03-02)")
        print("=" * 70)

        # ── Step 1: Load main page ──
        print("\n[1/4] Loading T-Rank page...")
        await page.goto(
            "https://barttorvik.com/trank.php?year=2026",
            wait_until="networkidle", timeout=60000,
        )
        await page.wait_for_timeout(5000)

        # ── Step 2: Download raw CSV ──
        print("[2/4] Downloading raw CSV...")
        async with page.expect_download(timeout=30000) as dl_info:
            await page.evaluate(
                "() => { window.location.href = "
                "'https://barttorvik.com/trank.php?year=2026&csv=1'; }"
            )
        download = await dl_info.value
        await download.save_as(str(RAW_CSV))
        print(f"  Saved raw CSV: {RAW_CSV}")

        # ── Step 3: Scrape conference data from DOM ──
        print("[3/4] Scraping conference data from DOM...")
        await page.goto(
            "https://barttorvik.com/trank.php?year=2026",
            wait_until="networkidle", timeout=60000,
        )
        await page.wait_for_timeout(5000)

        dom_data = await scrape_dom_table(page)
        conf_map = {}
        if dom_data and len(dom_data["rows"]) > 50:
            hdrs = dom_data["headers"]
            col_start = next((i for i, h in enumerate(hdrs) if h.strip() == "RK"), 0)
            stat_hdrs = [h.strip() for h in hdrs[col_start:]]
            team_idx = stat_hdrs.index("TEAM") if "TEAM" in stat_hdrs else None
            conf_idx = stat_hdrs.index("CONF") if "CONF" in stat_hdrs else None

            if team_idx is not None and conf_idx is not None:
                for row in dom_data["rows"]:
                    if len(row) > max(team_idx, conf_idx):
                        team = clean_team_name(row[team_idx])
                        conf = row[conf_idx].strip()
                        conf_map[team] = conf
                print(f"  Got conference for {len(conf_map)} teams")

        # ── Step 4: Scrape Elite SOS ──
        print("[4/4] Scraping Elite SOS...")
        await page.goto(
            "https://barttorvik.com/sos.php?year=2026",
            wait_until="networkidle", timeout=60000,
        )
        await page.wait_for_timeout(3000)

        sos_map = {}
        try:
            await page.wait_for_selector("table", timeout=10000)
            sos_data = await scrape_dom_table(page)
            if sos_data and len(sos_data["rows"]) > 50:
                hdrs = sos_data["headers"]
                team_idx = next((i for i, h in enumerate(hdrs) if "TEAM" in h.upper()), None)
                elite_indices = [i for i, h in enumerate(hdrs) if h.strip().upper() == "ELITE"]
                elite_idx = elite_indices[-1] if elite_indices else None

                if team_idx is not None and elite_idx is not None:
                    for row in sos_data["rows"]:
                        if len(row) > max(team_idx, elite_idx):
                            team = clean_team_name(row[team_idx])
                            val = row[elite_idx].strip().split()[0]
                            sos_map[team] = val
                    print(f"  Got Elite SOS for {len(sos_map)} teams")
        except Exception as e:
            print(f"  SOS scrape failed: {e}")

        await browser.close()

    # ── Process raw CSV ──
    print(f"\n{'=' * 70}")
    print("PROCESSING DATA")
    print("=" * 70)

    df_raw = pd.read_csv(RAW_CSV, header=None)
    print(f"Raw CSV: {df_raw.shape[0]} teams, {df_raw.shape[1]} columns")

    col_rename = {k: v for k, v in CSV_COLUMNS.items() if k < df_raw.shape[1]}
    df = df_raw.rename(columns=col_rename)

    drop_cols = [c for c in df.columns if str(c).startswith("_") or isinstance(c, int)]
    df = df.drop(columns=drop_cols)

    # Add conference from DOM
    df["Conf"] = df["Team"].map(conf_map)
    unmatched_conf = df["Conf"].isna().sum()
    if unmatched_conf > 0:
        print(f"  {unmatched_conf} teams missing conference from DOM")
        # Try matching by prefix for multi-word names
        for idx, row in df[df["Conf"].isna()].iterrows():
            csv_name = row["Team"]
            for dom_name, conf in conf_map.items():
                if csv_name.lower().startswith(dom_name.lower()) or \
                   dom_name.lower().startswith(csv_name.lower()):
                    df.at[idx, "Conf"] = conf
                    break

    unmatched_conf = df["Conf"].isna().sum()
    if unmatched_conf > 0:
        print(f"  Still {unmatched_conf} teams without conference after fuzzy match")

    # Add Elite SOS
    df["Elite SOS"] = df["Team"].map(sos_map)
    unmatched_sos = df["Elite SOS"].isna().sum()
    if unmatched_sos > 0:
        for idx, row in df[df["Elite SOS"].isna()].iterrows():
            csv_name = row["Team"]
            for dom_name, val in sos_map.items():
                if csv_name.lower().startswith(dom_name.lower()) or \
                   dom_name.lower().startswith(csv_name.lower()):
                    df.at[idx, "Elite SOS"] = val
                    break

    # Convert numeric columns
    num_cols = ["AdjOE", "AdjDE", "Barthag", "G", "W",
                "EFG%", "EFGD%", "FTR", "FTRD", "TOR", "TORD",
                "ORB", "DRB", "AdjT", "2P%", "2P%D", "3P%", "3P%D",
                "3PR", "3PRD", "WAB", "Hgt", "Elite SOS"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse record
    if "Rec" in df.columns:
        rec_split = df["Rec"].str.split("-", expand=True)
        df["L"] = pd.to_numeric(rec_split[1], errors="coerce")

    # Derived columns
    df["AdjEM"] = df["AdjOE"] - df["AdjDE"]
    df["YEAR"] = 2026

    # Rename Hgt to AVG HGT
    if "Hgt" in df.columns:
        df = df.rename(columns={"Hgt": "AVG HGT"})

    # Sort by AdjEM
    df = df.sort_values("AdjEM", ascending=False).reset_index(drop=True)

    # Select final columns
    final_cols = ["Team", "Conf", "G", "Rec", "W", "L",
                  "AdjOE", "AdjDE", "AdjEM", "Barthag", "AdjT",
                  "EFG%", "EFGD%", "TOR", "TORD", "ORB", "DRB",
                  "FTR", "FTRD", "2P%", "2P%D", "3P%", "3P%D",
                  "3PR", "3PRD", "Elite SOS", "WAB", "AVG HGT", "YEAR"]
    df = df[[c for c in final_cols if c in df.columns]]

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} teams to {OUTPUT}")
    print(f"Columns: {list(df.columns)}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("TOP 25 BY ADJUSTED EFFICIENCY MARGIN")
    print(f"{'=' * 70}")
    for i, (_, r) in enumerate(df.head(25).iterrows()):
        tal = ""
        conf = f"({r['Conf']:5})" if pd.notna(r.get("Conf")) else ""
        rec = f"{r['Rec']:>5}" if pd.notna(r.get("Rec")) else ""
        sos = f"SOS: {r['Elite SOS']:4.0f}" if pd.notna(r.get("Elite SOS")) else ""
        print(f"  {i+1:2}. {r['Team']:24} {conf} {rec} | "
              f"EM: {r['AdjEM']:+6.1f} | BARTHAG: {r['Barthag']:.4f} | {sos}")

    # Data quality
    print(f"\n{'=' * 70}")
    print("DATA QUALITY")
    print(f"{'=' * 70}")
    for c in df.columns:
        if c in ("Team", "Conf", "Rec", "YEAR"):
            continue
        valid = df[c].notna().sum()
        status = "OK" if valid == len(df) else f"{valid}/{len(df)}"
        print(f"  {c:12}: {status}")

    missing = {"Talent", "Exp"} - set(df.columns)
    if missing:
        print(f"\n  NOTE: {missing} not available in CSV download.")
        print("  These will be approximated from 2025 data during conversion.")


asyncio.run(fetch_all())
