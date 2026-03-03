"""
Stathead player data scraper for 2002-03 through 2006-07 NCAA seasons.

Uses Playwright browser automation — logs in once, then walks through each
paginated results page (200 rows each) extracting the table, with polite delays.

Usage:
    python fetch_stathead_fast.py --email EMAIL --password PASS
    python fetch_stathead_fast.py --process-only   (re-process already scraped data)
"""
import asyncio
import argparse
import re
import sys
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STATHEAD_DIR = DATA_DIR / "historical" / "stathead"
STATHEAD_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = STATHEAD_DIR / "download.log"

CLASS_MAP = {
    "fr": 0, "freshman": 0,
    "so": 1, "sophomore": 1,
    "jr": 2, "junior": 2,
    "sr": 3, "senior": 3,
    "gr": 3, "graduate": 3,
    "rs fr": 0, "rs so": 1, "rs jr": 2, "rs sr": 3,
}

# Login page — stathead.com redirects to sports-reference for auth
LOGIN_URL = "https://stathead.com/users/login.cgi"

# Base query URL — NCAA Men, 2002-03 to 2006-07, sorted by minutes played
BASE_QUERY = (
    "https://www.sports-reference.com/stathead/basketball/cbb/player-season-finder.cgi"
    "?request=1"
    "&comp_id=NCAAM"
    "&phase_id=1"
    "&match=player_season"
    "&year_min=2003"
    "&year_max=2007"
    "&display_type=totals"
    "&order_by=mp"
    "&draft_pick_type=overall"
)


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def parse_height(h):
    if not h or str(h).strip() in ("", "-", "nan"):
        return None
    h = str(h).strip()
    m = re.match(r"(\d+)-(\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    m = re.match(r"(\d+)[''`](\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return None


# ---------------------------------------------------------------------------
# Playwright scraper
# ---------------------------------------------------------------------------

async def scrape_all_pages_manual(email: str) -> pd.DataFrame:
    """
    Open a visible browser, navigate to Stathead login, wait for the user
    to log in manually, then scrape all result pages automatically.
    """
    from playwright.async_api import async_playwright

    all_rows = []
    col_headers = None
    progress_file = STATHEAD_DIR / "scrape_progress.csv"

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-first-run"],
        )
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            accept_downloads=True,
        )
        await ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )
        page = await ctx.new_page()

        # Navigate to Stathead login page (has Google Sign-In button)
        log("[Login] Opening Stathead login page...")
        log("[Login] *** BROWSER WINDOW IS OPENING ON YOUR SCREEN ***")
        log("[Login] Please click 'Sign in with Google' and complete the Google login.")
        log(f"[Login] Use Google account: {email}")
        await page.goto("https://stathead.com/users/login.cgi", wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)
        log(f"[Login] Page loaded: {page.url}")

        # Try to find and click the Google sign-in button
        google_clicked = False
        for sel in [
            'a[href*="google"]',
            'button:has-text("Google")',
            'a:has-text("Google")',
            '[data-provider="google"]',
            '.google-login',
            'a[href*="oauth"]',
        ]:
            try:
                cnt = await page.locator(sel).count()
                if cnt > 0:
                    await page.locator(sel).first.click()
                    log(f"[Login] Clicked Google sign-in button ({sel})")
                    google_clicked = True
                    break
            except Exception:
                pass

        if not google_clicked:
            log("[Login] Could not auto-click Google button — please click it manually.")
            # Show what buttons/links are on the page
            links = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a, button'))
                    .map(el => ({text: (el.innerText||'').trim().slice(0,40), href: (el.href||'').slice(0,80)}))
                    .filter(x => x.text.length > 0)
                    .slice(0, 20);
            }""")
            log("[Login] Links/buttons on page:")
            for lnk in links:
                log(f"  {lnk['text']!r}  {lnk['href']!r}")

        # Wait up to 4 minutes for Google OAuth to complete
        log("\n[Login] Waiting for you to complete Google sign-in (up to 4 minutes)...")
        log("[Login] The browser window should be visible — check your taskbar.")
        for i in range(48):  # 48 * 5s = 4 minutes
            await page.wait_for_timeout(5000)
            try:
                body = await page.inner_text("body")
                url = page.url
            except Exception:
                continue
            if "logout" in body.lower() or "log out" in body.lower() or "sign out" in body.lower():
                log(f"[Login] SUCCESS — logged in! URL: {url}")
                break
            # Also check if we're on a non-login page (Google OAuth complete redirected us)
            if "stathead.com" in url and "login" not in url and "google" not in url:
                log(f"[Login] Redirected to Stathead page — likely logged in: {url}")
                break
            if i % 6 == 5:
                elapsed = (i + 1) * 5
                log(f"[Login] Still waiting... ({elapsed}s / 240s)")
        else:
            log("[Login] Timeout — proceeding (may still work if cookies were set)")

        # Now scrape pages
        log("\n[2/3] Starting page scrape...")
        offset = 0
        page_num = 0
        max_pages = 30

        while page_num < max_pages:
            url = BASE_QUERY + f"&offset={offset}"
            log(f"\n  Page {page_num + 1} (offset={offset})")

            for attempt in range(3):
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    break
                except Exception as e:
                    log(f"  Attempt {attempt+1} failed: {e}")
                    await page.wait_for_timeout(5000)
            else:
                log("  All attempts failed. Stopping.")
                break

            await page.wait_for_timeout(4000)

            if "login" in page.url.lower():
                log("  Redirected to login — session lost. Stopping.")
                break

            table_html = await page.evaluate("""() => {
                const sels = ['#div_results table','#results table','table#results','.table-wrap table','table'];
                for (const s of sels) {
                    const t = document.querySelector(s);
                    if (t) return t.outerHTML;
                }
                return null;
            }""")

            if not table_html:
                vis = await page.evaluate("() => document.body.innerText.slice(0,300)")
                log(f"  No table. Page text: {vis[:200]}")
                break

            try:
                dfs = pd.read_html(table_html)
                if not dfs:
                    break
                tdf = dfs[0]
            except Exception as e:
                log(f"  Table parse error: {e}")
                break

            if col_headers is None:
                col_headers = list(tdf.columns)
                log(f"  Headers: {col_headers[:15]}")

            rk_col = [c for c in tdf.columns if str(c).strip().lower() == "rk"]
            if rk_col:
                tdf = tdf[tdf[rk_col[0]].astype(str) != "Rk"]

            rows_this_page = len(tdf)
            log(f"  Rows: {rows_this_page} (total so far: {sum(len(d) for d in all_rows) + rows_this_page})")

            if rows_this_page == 0:
                break

            all_rows.append(tdf)
            combined = pd.concat(all_rows, ignore_index=True)
            combined.to_csv(progress_file, index=False)

            if rows_this_page < 200:
                log("  Last page. Done.")
                break

            offset += 200
            page_num += 1
            log(f"  Waiting 6s...")
            await page.wait_for_timeout(6000)

        await browser.close()

    if not all_rows:
        log("\nERROR: No data collected!")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    log(f"\n[Scrape] Total rows: {len(df)}")
    raw_path = STATHEAD_DIR / "players_2002_2006_raw.csv"
    df.to_csv(raw_path, index=False)
    log(f"[Scrape] Saved: {raw_path}")
    return df


async def scrape_all_pages(email: str, passwords: list) -> pd.DataFrame:
    """Login to Stathead and scrape all result pages."""
    from playwright.async_api import async_playwright

    all_rows = []
    col_headers = None
    progress_file = STATHEAD_DIR / "scrape_progress.csv"

    async with async_playwright() as p:
        # Use VISIBLE browser — Cloudflare blocks headless
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            accept_downloads=True,
        )

        # Remove the webdriver flag that Cloudflare looks for
        await ctx.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)

        page = await ctx.new_page()

        # ---- Login ----
        log("[1/3] Logging in to Stathead...")
        await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
        # Give Cloudflare time to resolve (it usually passes in visible browser)
        await page.wait_for_timeout(5000)
        log(f"  Login page URL: {page.url}")

        # Check if Cloudflare challenge is present; wait up to 20s for it to clear
        for cf_wait in range(8):
            body_cf = await page.inner_text("body")
            if "security verification" in body_cf.lower() or "ray id" in body_cf.lower():
                log(f"  Cloudflare challenge detected — waiting ({cf_wait+1}/8)...")
                await page.wait_for_timeout(3000)
            else:
                break

        log(f"  Page after CF wait: {page.url}")

        login_success = False
        for attempt, password in enumerate(passwords):
            log(f"\n  Trying password #{attempt+1}...")

            # Re-navigate if needed
            if "login" not in page.url.lower() and attempt > 0:
                await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(3000)

            # Fill username
            for sel in ['#username', 'input[name="username"]', 'input[type="email"]',
                        'input[placeholder*="email" i]']:
                try:
                    if await page.locator(sel).count() > 0:
                        await page.locator(sel).first.clear()
                        await page.locator(sel).first.fill(email)
                        log(f"  Email -> {sel}")
                        break
                except Exception:
                    pass

            await page.wait_for_timeout(400)

            # Fill password
            for sel in ['#password', 'input[name="password"]', 'input[type="password"]']:
                try:
                    if await page.locator(sel).count() > 0:
                        await page.locator(sel).first.clear()
                        await page.locator(sel).first.fill(password)
                        log(f"  Password -> {sel}")
                        break
                except Exception:
                    pass

            await page.wait_for_timeout(400)

            # Submit
            for sel in ['button[type="submit"]', 'input[type="submit"]',
                        'input[value="Login"]', '#login-button', '.btn-primary']:
                try:
                    if await page.locator(sel).count() > 0:
                        await page.locator(sel).first.click()
                        log(f"  Clicked: {sel}")
                        break
                except Exception:
                    pass

            # Wait for redirect
            await page.wait_for_timeout(6000)
            post_url = page.url
            body_text = await page.inner_text("body")
            log(f"  Post-submit URL: {post_url}")

            if "log out" in body_text.lower() or "logout" in body_text.lower():
                log(f"  LOGIN SUCCESS with password #{attempt+1}")
                login_success = True
                break
            elif "login" in post_url.lower():
                log(f"  Password #{attempt+1} failed — still on login page")
                if "incorrect" in body_text.lower() or "invalid" in body_text.lower():
                    log("  Server says: incorrect credentials")
            else:
                log(f"  Post-submit URL changed — likely logged in: {post_url}")
                login_success = True
                break

        if not login_success:
            log("\nWARNING: Could not confirm login — attempting to navigate anyway...")
            log("(Browser may be open — check if you need to complete a CAPTCHA)")
            # Wait a bit more in case user needs to interact
            await page.wait_for_timeout(8000)

        # ---- Navigate to query ----
        log("\n[2/3] Navigating to player season finder...")
        offset = 0
        page_num = 0
        max_pages = 30  # Safety ceiling (30 * 200 = 6000 rows max)

        while page_num < max_pages:
            url = BASE_QUERY + f"&offset={offset}"
            log(f"\n  Page {page_num + 1} (offset={offset}): {url}")

            # Navigate with retry
            for attempt in range(3):
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    break
                except Exception as e:
                    log(f"  Attempt {attempt+1} failed: {e}")
                    await page.wait_for_timeout(5000)
            else:
                log("  All navigation attempts failed. Stopping.")
                break

            # Wait for content
            await page.wait_for_timeout(4000)

            # Check if we hit a login wall or error
            cur_url = page.url
            content = await page.content()

            if "login" in cur_url.lower() or "sign in" in cur_url.lower():
                log("  Redirected to login — session expired. Stopping.")
                break

            if "subscribe" in content.lower() and len(content) < 3000:
                log("  Hit subscription wall. Stopping.")
                break

            # Find the results table
            table_html = await page.evaluate("""() => {
                // Look for the main stats table
                const selectors = [
                    '#div_results table',
                    '#results table',
                    'table#results',
                    '.table-wrap table',
                    'table.stats_table',
                    'table',
                ];
                for (const sel of selectors) {
                    const t = document.querySelector(sel);
                    if (t) return t.outerHTML;
                }
                return null;
            }""")

            if not table_html:
                log("  No table found on page. Checking page content...")
                # Debug: show first 500 chars of visible text
                vis = await page.evaluate("() => document.body.innerText.slice(0, 500)")
                log(f"  Page text preview: {vis[:300]}")
                log("  Stopping — no more data.")
                break

            # Parse table with pandas
            try:
                dfs = pd.read_html(table_html)
                if not dfs:
                    log("  Empty table parse. Stopping.")
                    break
                tdf = dfs[0]
                log(f"  Table shape: {tdf.shape}")
            except Exception as e:
                log(f"  Table parse error: {e}")
                break

            # Get headers from first page
            if col_headers is None:
                col_headers = list(tdf.columns)
                log(f"  Headers: {col_headers[:15]}")

            # Remove header rows embedded in data (where Rk == 'Rk')
            rk_col = [c for c in tdf.columns if str(c).strip().lower() == "rk"]
            if rk_col:
                tdf = tdf[tdf[rk_col[0]].astype(str) != "Rk"]

            rows_this_page = len(tdf)
            log(f"  Rows on this page: {rows_this_page}")

            if rows_this_page == 0:
                log("  Empty page. Done.")
                break

            all_rows.append(tdf)

            # Save progress incrementally
            combined = pd.concat(all_rows, ignore_index=True)
            combined.to_csv(progress_file, index=False)

            if rows_this_page < 200:
                log("  Last page (fewer than 200 rows). Done.")
                break

            offset += 200
            page_num += 1

            # Polite delay between pages
            delay = 5 if page_num < 5 else 8
            log(f"  Waiting {delay}s before next page...")
            await page.wait_for_timeout(delay * 1000)

        await browser.close()

    if not all_rows:
        log("\nERROR: No data collected!")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    log(f"\n[Scrape] Total rows: {len(df)}")

    # Save raw scraped data
    raw_path = STATHEAD_DIR / "players_2002_2006_raw.csv"
    df.to_csv(raw_path, index=False)
    log(f"[Scrape] Raw data saved: {raw_path}")
    return df


# ---------------------------------------------------------------------------
# Process player data into team-level metrics
# ---------------------------------------------------------------------------

def process_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw player rows to team-level EXP and AVG HGT."""
    log("\n[Process] Processing player data...")

    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]", "_", regex=True)
    log(f"  Columns: {list(df.columns[:20])}")

    aliases = {
        "school": "TEAM", "team": "TEAM",
        "season": "YEAR",
        "class": "CLASS", "class_": "CLASS", "yr": "CLASS", "cl": "CLASS",
        "height": "HT", "ht": "HT",
        "mp": "MP", "min": "MP", "minutes_played": "MP",
        "g": "G", "games": "G",
        "player": "PLAYER",
    }
    rename = {old: new for old, new in aliases.items() if old in df.columns and new not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    log(f"  Renamed: {rename}")

    if "TEAM" not in df.columns:
        log(f"  ERROR: No TEAM column. Available: {list(df.columns)}")
        return pd.DataFrame()

    if "YEAR" not in df.columns:
        log(f"  ERROR: No YEAR column. Available: {list(df.columns)}")
        return pd.DataFrame()

    # Parse year: "2002-03" -> 2003
    year_series = df["YEAR"].astype(str)
    extracted = year_series.str.extract(r"(\d{4})-")[0]
    if extracted.notna().sum() > 0:
        df["YEAR"] = pd.to_numeric(extracted, errors="coerce") + 1
    else:
        df["YEAR"] = pd.to_numeric(year_series.str.extract(r"(\d{4})")[0], errors="coerce")

    df = df.dropna(subset=["TEAM", "YEAR"])
    df["YEAR"] = df["YEAR"].astype(int)
    log(f"  Years: {sorted(df['YEAR'].unique())}")
    log(f"  Teams: {df['TEAM'].nunique()} unique schools")

    # Parse class
    if "CLASS" in df.columns:
        df["class_val"] = (
            df["CLASS"].astype(str).str.strip().str.lower()
            .map(lambda x: CLASS_MAP.get(x, np.nan))
        )
    else:
        df["class_val"] = np.nan

    # Parse height
    if "HT" in df.columns:
        df["height_in"] = df["HT"].apply(parse_height)
    else:
        df["height_in"] = None

    # Minutes
    if "MP" in df.columns:
        df["minutes"] = pd.to_numeric(df["MP"], errors="coerce").fillna(0)
    else:
        df["minutes"] = 100.0

    log(f"  Class coverage: {df['class_val'].notna().sum()}/{len(df)} ({df['class_val'].notna().mean()*100:.0f}%)")
    if "height_in" in df.columns:
        log(f"  Height coverage: {pd.Series(df['height_in']).notna().sum()}/{len(df)}")

    results = []
    for (team, year), grp in df.groupby(["TEAM", "YEAR"]):
        wt = grp[grp["minutes"] > 0].copy()
        if len(wt) == 0:
            wt = grp.copy()
            wt["minutes"] = 100.0

        exp_d = wt[wt["class_val"].notna()]
        exp = (
            (exp_d["class_val"] * exp_d["minutes"]).sum() / exp_d["minutes"].sum()
            if len(exp_d) > 0 and exp_d["minutes"].sum() > 0 else np.nan
        )

        hgt_d = wt[pd.Series(wt["height_in"]).notna().values] if "height_in" in wt.columns else pd.DataFrame()
        avg_hgt = (
            (hgt_d["height_in"] * hgt_d["minutes"]).sum() / hgt_d["minutes"].sum()
            if len(hgt_d) > 0 and hgt_d["minutes"].sum() > 0 else np.nan
        )

        results.append({
            "TEAM": team,
            "YEAR": int(year),
            "EXP": round(float(exp), 3) if pd.notna(exp) else None,
            "AVG HGT": round(float(avg_hgt), 1) if pd.notna(avg_hgt) else None,
            "n_players": len(grp),
        })

    result_df = pd.DataFrame(results)
    log(f"\n[Process] Team-years computed: {len(result_df)}")

    metrics_path = STATHEAD_DIR / "team_metrics_combined.csv"
    result_df.to_csv(metrics_path, index=False)
    log(f"[Process] Saved: {metrics_path}")
    return result_df


# ---------------------------------------------------------------------------
# Merge into main dataset
# ---------------------------------------------------------------------------

def merge_into_main(metrics_df: pd.DataFrame):
    main_path = RAW_DIR / "KenPom Barttorvik.csv"
    df = pd.read_csv(main_path)

    def normalize(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

    kp_norm = {normalize(t): t for t in df["TEAM"].unique()}

    name_map = {}
    for sh_name in set(metrics_df["TEAM"].unique()) - set(df["TEAM"].unique()):
        n = normalize(sh_name)
        if n in kp_norm:
            name_map[sh_name] = kp_norm[n]
        else:
            sh_words = set(n.split())
            best, best_score = None, 0
            for kp_n, kp_t in kp_norm.items():
                kp_words = set(kp_n.split())
                if not sh_words or not kp_words:
                    continue
                score = len(sh_words & kp_words) / max(len(sh_words | kp_words), 1)
                if score > best_score and score > 0.6:
                    best, best_score = kp_t, score
            if best:
                name_map[sh_name] = best

    filled_exp = filled_hgt = 0
    # Stathead "2002-03" season → YEAR 2003; covers 2003-2007 in our dataset
    target_years = [2003, 2004, 2005, 2006, 2007]
    for _, row in metrics_df.iterrows():
        if int(row["YEAR"]) not in target_years:
            continue
        kp_name = name_map.get(row["TEAM"], row["TEAM"])
        mask = (df["TEAM"] == kp_name) & (df["YEAR"] == int(row["YEAR"]))
        if not mask.any():
            continue

        # Always overwrite era-average placeholders with real Stathead data
        if pd.notna(row.get("EXP")):
            df.loc[mask, "EXP"] = row["EXP"]
            filled_exp += 1

        if pd.notna(row.get("AVG HGT")) and (
            pd.isna(df.loc[mask, "AVG HGT"]).all() or (df.loc[mask, "AVG HGT"] < 70).all()
        ):
            df.loc[mask, "AVG HGT"] = row["AVG HGT"]
            filled_hgt += 1

    log(f"\n[Merge] Filled EXP: {filled_exp}, AVG HGT: {filled_hgt}")

    # Fill remaining gaps with era averages
    era_exp = df[(df["YEAR"].between(2007, 2010)) & df["EXP"].notna()]["EXP"].mean()
    era_hgt = df[(df["YEAR"].between(2007, 2010)) & (df["AVG HGT"] > 70)]["AVG HGT"].mean()

    miss_exp = df["EXP"].isna() & df["YEAR"].isin(target_years)
    miss_hgt = (df["AVG HGT"].isna() | (df["AVG HGT"] < 70)) & df["YEAR"].isin(target_years)
    df.loc[miss_exp, "EXP"] = round(era_exp, 3)
    df.loc[miss_hgt, "AVG HGT"] = round(era_hgt, 1)

    log(f"[Merge] Era defaults: EXP={era_exp:.3f}, HGT={era_hgt:.1f}")
    df.to_csv(main_path, index=False)

    log("\n" + "=" * 50)
    log("FINAL COVERAGE")
    log("=" * 50)
    for col in ["EXP", "AVG HGT", "ELITE SOS"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            log(f"  {col:12}: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Clear log file
    LOG_FILE.write_text("", encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str)
    parser.add_argument("--password", type=str)
    parser.add_argument("--process-only", action="store_true")
    args = parser.parse_args()

    if args.process_only:
        raw_csv = STATHEAD_DIR / "players_2002_2006_raw.csv"
        progress_csv = STATHEAD_DIR / "scrape_progress.csv"
        if raw_csv.exists():
            log(f"[Main] Processing: {raw_csv}")
            df_raw = pd.read_csv(raw_csv)
        elif progress_csv.exists():
            log(f"[Main] Processing progress file: {progress_csv}")
            df_raw = pd.read_csv(progress_csv)
        else:
            log(f"No raw data found in {STATHEAD_DIR}")
            return
        metrics = process_player_data(df_raw)
        if not metrics.empty:
            merge_into_main(metrics)
            log("\n[Main] Rebuilding extended dataset...")
            subprocess.run(["python", "run_reconstruction.py", "--run"], cwd=str(PROJECT_ROOT))
        return

    if not args.email or not args.password:
        log("Usage: python fetch_stathead_fast.py --email EMAIL --password PASS")
        return

    log(f"[Main] Starting Stathead scraper (manual login mode)")
    log(f"[Main] A browser window will open — please log in to Stathead.")
    log(f"[Main] Target: {BASE_QUERY}")

    df_raw = asyncio.run(scrape_all_pages_manual(args.email))

    if df_raw.empty:
        log("\nNo data collected. Possible fixes:")
        log("  1. Check credentials")
        log("  2. Try logging into stathead.com manually first")
        log("  3. Check if Sports-Reference is accessible")
        return

    metrics = process_player_data(df_raw)
    if not metrics.empty:
        merge_into_main(metrics)
        log("\n[Main] Rebuilding extended dataset...")
        subprocess.run(["python", "run_reconstruction.py", "--run"], cwd=str(PROJECT_ROOT))
        log("\nDONE!")


if __name__ == "__main__":
    main()
