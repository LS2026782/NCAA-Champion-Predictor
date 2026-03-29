"""
Comprehensive data collection for missing features.
Collects real data for ELITE SOS, TALENT, EXP, and HGT.

Phases (run independently or together):
  --phase1   Barttorvik Elite SOS (all years, ~5 min)
  --phase2   247Sports Team Talent Composite (~5 min)
  --phase3   Sports-Reference rosters for EXP/HGT 2002-2006 (~20 min)
  --integrate Merge all collected data into main dataset
"""

import asyncio
import re
import time
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests as req_lib
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HIST_DIR = DATA_DIR / "historical"
BART_DIR = HIST_DIR / "barttorvik"
TALENT_DIR = HIST_DIR / "talent"
ROSTER_DIR = HIST_DIR / "rosters"

for d in [BART_DIR, TALENT_DIR, ROSTER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SKIP_YEAR = 2020


# ============================================================================
# PHASE 1: Barttorvik Elite SOS
# ============================================================================

async def scrape_barttorvik_table(page):
    """Extract all data from the largest table on a Barttorvik page."""
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
            if (cells.length > 3) rows.push(cells);
        });
        return { headers: hdrs, rows: rows };
    }""")
    return data


def clean_team(raw):
    """Strip game annotations from team name."""
    m = re.match(r"^(.+)\s+\([AHN]\)\s+\d+\s", raw)
    return m.group(1).strip() if m else raw.strip()


def parse_numeric(val):
    """Safely parse a numeric value from table cell text."""
    if not val or val == '-':
        return np.nan
    val = val.strip().split()[0]
    try:
        return float(val)
    except ValueError:
        return np.nan


def clean_sos_team(raw):
    """Strip trailing seed number from SOS page team name."""
    return re.sub(r"\s+\d+$", "", raw.strip())


async def fetch_elite_sos_all_years(start_year=2002, end_year=2025):
    """Scrape Barttorvik SOS page for Elite SOS, all years."""
    from playwright.async_api import async_playwright

    all_data = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
        )
        page = await ctx.new_page()

        years = [y for y in range(start_year, end_year + 1) if y != SKIP_YEAR]
        total = len(years)

        print("=" * 70)
        print(f"PHASE 1: Fetching Elite SOS from Barttorvik ({total} years)")
        print("=" * 70)

        for i, year in enumerate(years):
            print(f"\n[{i+1}/{total}] Year {year}...", end=" ")
            url = f"https://barttorvik.com/sos.php?year={year}"

            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
                await page.wait_for_timeout(4000)

                final_url = page.url
                if "trank.php" in final_url:
                    print(f"Redirected to T-Rank - SOS page not available for {year}")
                    continue

                await page.wait_for_selector("table", timeout=10000)

                table_data = await scrape_barttorvik_table(page)
                if not table_data or len(table_data["rows"]) < 50:
                    print(f"Only {len(table_data['rows']) if table_data else 0} rows")
                    continue

                # SOS page structure (verified via debug):
                # Headers: ['', '', 'NON-CON', 'OVERALL', 'RK', 'TEAM', 'CONF',
                #           'ELITE', 'BASIC', 'CURRENT', 'ELITE', 'BASIC', 'CURRENT']
                # Row data: [RK, TEAM, CONF, NC_ELITE, NC_BASIC, NC_CURRENT,
                #            OVR_ELITE, OVR_BASIC, OVR_CURRENT]
                # TEAM has trailing seed (e.g., "Alabama 4")
                # ELITE values are "XX % rank" format (e.g., "38 % 1")

                year_records = []
                for row in table_data["rows"]:
                    if len(row) < 7:
                        continue

                    team_raw = row[1].strip()
                    team = clean_sos_team(team_raw)
                    elite_str = row[6].strip()

                    elite_val = parse_numeric(elite_str)
                    if team and not pd.isna(elite_val):
                        year_records.append({
                            "TEAM": team,
                            "ELITE SOS": elite_val,
                            "YEAR": year,
                        })

                if year_records:
                    df_year = pd.DataFrame(year_records)
                    df_year.to_csv(BART_DIR / f"elite_sos_{year}.csv", index=False)
                    all_data.extend(year_records)
                    print(f"Got {len(year_records)} teams")
                else:
                    print("No data parsed")
                    if table_data["rows"]:
                        print(f"    Sample row: {table_data['rows'][0][:10]}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        await browser.close()

    # For years without SOS page (2002-2007), approximate from KenPom SOS
    scraped_years = {r["YEAR"] for r in all_data}
    missing_years = [y for y in range(start_year, end_year + 1)
                     if y != SKIP_YEAR and y not in scraped_years]

    if missing_years:
        print(f"\nApproximating Elite SOS for years without Barttorvik SOS page: {missing_years}")
        main_df = pd.read_csv(RAW_DIR / "KenPom Barttorvik.csv")

        for year in missing_years:
            year_df = main_df[main_df["YEAR"] == year].copy()
            if "SOS" not in year_df.columns or year_df["SOS"].isna().all():
                print(f"  {year}: No SOS data available for approximation")
                continue

            sos_vals = year_df["SOS"].dropna()
            sos_min, sos_max = sos_vals.min(), sos_vals.max()
            if sos_max <= sos_min:
                continue

            year_records = []
            for _, row in year_df.iterrows():
                if pd.notna(row["SOS"]):
                    scaled = ((row["SOS"] - sos_min) / (sos_max - sos_min)) * 40
                    year_records.append({
                        "TEAM": row["TEAM"],
                        "ELITE SOS": round(scaled, 1),
                        "YEAR": year,
                    })

            if year_records:
                df_year = pd.DataFrame(year_records)
                df_year.to_csv(BART_DIR / f"elite_sos_{year}.csv", index=False)
                all_data.extend(year_records)
                print(f"  {year}: Approximated for {len(year_records)} teams")

    if all_data:
        df_all = pd.DataFrame(all_data)
        df_all.to_csv(BART_DIR / "elite_sos_all.csv", index=False)
        print(f"\nSaved {len(df_all)} total Elite SOS records to {BART_DIR / 'elite_sos_all.csv'}")

        coverage = df_all.groupby("YEAR").size()
        print("\nCoverage by year:")
        for yr, cnt in coverage.items():
            print(f"  {yr}: {cnt} teams")

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


# ============================================================================
# PHASE 2: 247Sports Team Talent Composite
# ============================================================================

async def fetch_247_talent_page(page, year):
    """
    Scrape 247Sports Team Talent Composite for a single year.
    Returns list of {team, talent_score, year} dicts.
    """
    url = f"https://247sports.com/Season/{year}-Basketball/CollegeTeamTalentComposite/"
    records = []

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        talent_data = await page.evaluate(r"""() => {
            const results = [];
            // 247Sports uses ranking tables with various class structures
            const rows = document.querySelectorAll('.rankings-page__list-item, .team-rankings-item, tr');
            for (const row of rows) {
                // Try to find team name and talent score
                const teamEl = row.querySelector('.team-name, .name a, td:nth-child(2) a, .rankings-page__name-link');
                const scoreEl = row.querySelector('.talent-composite, .score, td:last-child, .rankings-page__points');
                const rankEl = row.querySelector('.rank-column, .primary, td:first-child, .rankings-page__rank');
                
                if (teamEl) {
                    const team = teamEl.innerText.trim();
                    const score = scoreEl ? scoreEl.innerText.trim() : '';
                    const rank = rankEl ? rankEl.innerText.trim() : '';
                    if (team && team.length > 1 && team.length < 50) {
                        results.push({ team, score, rank });
                    }
                }
            }
            
            // Also try composite talent table format
            if (results.length < 10) {
                const listItems = document.querySelectorAll('.composite-team li, .rankings-list li');
                for (const li of listItems) {
                    const team = li.querySelector('.name, .team')?.innerText?.trim() || '';
                    const score = li.querySelector('.score, .points, .rating')?.innerText?.trim() || '';
                    if (team) results.push({ team, score, rank: '' });
                }
            }

            // Fallback: try the main content area
            if (results.length < 10) {
                const mainTable = document.querySelector('table.rankings-table, .rankings-page__list');
                if (mainTable) {
                    const allRows = mainTable.querySelectorAll('tr, li, .rankings-page__list-item');
                    for (const r of allRows) {
                        const cells = r.querySelectorAll('td, .cell, span');
                        if (cells.length >= 2) {
                            results.push({
                                team: cells[1]?.innerText?.trim() || '',
                                score: cells[cells.length - 1]?.innerText?.trim() || '',
                                rank: cells[0]?.innerText?.trim() || '',
                            });
                        }
                    }
                }
            }

            return results;
        }""")

        if talent_data:
            for item in talent_data:
                team = item.get("team", "").strip()
                score_str = item.get("score", "").strip()
                if not team or len(team) < 2:
                    continue
                score = parse_numeric(score_str)
                records.append({
                    "TEAM": team,
                    "TALENT_SCORE": score,
                    "TALENT_RANK": item.get("rank", ""),
                    "YEAR": year,
                })

    except Exception as e:
        logger.warning(f"Error fetching 247Sports talent for {year}: {e}")

    return records


async def fetch_247_recruit_rankings(page, year, max_players=150):
    """
    Scrape 247Sports Composite recruit rankings for individual players.
    This is the fallback for years without Team Talent Composite.
    """
    url = f"https://247sports.com/Season/{year}-Basketball/CompositeRecruitRankings/?InstitutionGroup=HighSchool"
    records = []

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        recruit_data = await page.evaluate(r"""() => {
            const results = [];
            const items = document.querySelectorAll('.ri-page__list-item, .rankings-page__list-item');
            for (const item of items) {
                const rankEl = item.querySelector('.rank-column .primary, .rankings-page__rank-number');
                const nameEl = item.querySelector('.rankings-page__name-link, .player a');
                const schoolEl = item.querySelector('.meta .img-link, .school-name, .status .commit');
                const ratingEl = item.querySelector('.score, .rankings-page__star-and-score .score');
                
                if (nameEl) {
                    results.push({
                        rank: rankEl ? parseInt(rankEl.innerText.trim()) : null,
                        name: nameEl.innerText.trim(),
                        school: schoolEl ? schoolEl.innerText.trim() : '',
                        rating: ratingEl ? ratingEl.innerText.trim() : '',
                    });
                }
            }
            return results;
        }""")

        if recruit_data:
            for item in recruit_data[:max_players]:
                if item.get("name"):
                    records.append({
                        "year": year,
                        "rank": item.get("rank"),
                        "name": item["name"],
                        "school": item.get("school", ""),
                        "rating": item.get("rating", ""),
                    })

    except Exception as e:
        logger.warning(f"Error fetching 247Sports recruits for {year}: {e}")

    return records


async def fetch_talent_all_years(start_year=2003, end_year=2025):
    """Collect talent data from 247Sports for all years."""
    from playwright.async_api import async_playwright

    all_talent = []
    all_recruits = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
        )
        page = await ctx.new_page()

        years = [y for y in range(start_year, end_year + 1) if y != SKIP_YEAR]
        total = len(years)

        print("\n" + "=" * 70)
        print(f"PHASE 2: Fetching Talent from 247Sports ({total} years)")
        print("=" * 70)

        # Phase 2a: Try Team Talent Composite for each year
        print("\n--- Phase 2a: Team Talent Composite ---")
        for i, year in enumerate(years):
            print(f"[{i+1}/{total}] Team Talent {year}...", end=" ")
            records = await fetch_247_talent_page(page, year)
            if records and len(records) >= 20:
                df_year = pd.DataFrame(records)
                df_year.to_csv(TALENT_DIR / f"talent_{year}.csv", index=False)
                all_talent.extend(records)
                print(f"Got {len(records)} teams")
            else:
                print(f"Only {len(records)} records - will try recruit rankings")

            await page.wait_for_timeout(2000)

        # Phase 2b: For years missing team talent, get individual recruit rankings
        talent_years = {r["YEAR"] for r in all_talent if len([x for x in all_talent if x["YEAR"] == r["YEAR"]]) >= 20}
        missing_years = [y for y in years if y not in talent_years]

        if missing_years:
            print(f"\n--- Phase 2b: Individual Recruit Rankings ({len(missing_years)} years) ---")
            for i, year in enumerate(missing_years):
                recruit_year = year
                print(f"[{i+1}/{len(missing_years)}] Recruits class of {recruit_year}...", end=" ")
                records = await fetch_247_recruit_rankings(page, recruit_year)
                if records:
                    df_recruits = pd.DataFrame(records)
                    df_recruits.to_csv(TALENT_DIR / f"recruits_{recruit_year}.csv", index=False)
                    all_recruits.extend(records)
                    print(f"Got {len(records)} recruits")
                else:
                    print("No data")
                await page.wait_for_timeout(3000)

        await browser.close()

    if all_talent:
        df_talent = pd.DataFrame(all_talent)
        df_talent.to_csv(TALENT_DIR / "talent_all.csv", index=False)
        print(f"\nSaved {len(df_talent)} team talent records")

    if all_recruits:
        df_recruits = pd.DataFrame(all_recruits)
        df_recruits.to_csv(TALENT_DIR / "recruits_all.csv", index=False)
        print(f"Saved {len(df_recruits)} individual recruit records")

    return all_talent, all_recruits


# ============================================================================
# PHASE 3: Sports-Reference Rosters for EXP/HGT (2002-2006)
# ============================================================================

EXTENDED_TEAM_SLUGS = {
    'Air Force': 'air-force',
    'Akron': 'akron',
    'Alabama': 'alabama',
    'Alabama A&M': 'alabama-am',
    'Alabama St.': 'alabama-state',
    'Albany': 'albany-ny',
    'Alcorn St.': 'alcorn-state',
    'American': 'american',
    'Appalachian St.': 'appalachian-state',
    'Arizona': 'arizona',
    'Arizona St.': 'arizona-state',
    'Arkansas': 'arkansas',
    'Arkansas St.': 'arkansas-state',
    'Arkansas Pine Bluff': 'arkansas-pine-bluff',
    'Army': 'army',
    'Auburn': 'auburn',
    'Austin Peay': 'austin-peay',
    'BYU': 'brigham-young',
    'Ball St.': 'ball-state',
    'Baylor': 'baylor',
    'Belmont': 'belmont',
    'Bethune Cookman': 'bethune-cookman',
    'Binghamton': 'binghamton',
    'Boise St.': 'boise-state',
    'Boston College': 'boston-college',
    'Boston University': 'boston-university',
    'Bowling Green': 'bowling-green-state',
    'Bradley': 'bradley',
    'Brown': 'brown',
    'Bucknell': 'bucknell',
    'Buffalo': 'buffalo',
    'Butler': 'butler',
    'Cal Poly': 'cal-poly',
    'Cal St. Fullerton': 'cal-state-fullerton',
    'Cal St. Northridge': 'cal-state-northridge',
    'California': 'california',
    'Campbell': 'campbell',
    'Canisius': 'canisius',
    'Central Connecticut': 'central-connecticut-state',
    'Central Michigan': 'central-michigan',
    'Charleston': 'college-of-charleston',
    'Charleston Southern': 'charleston-southern',
    'Charlotte': 'charlotte',
    'Chattanooga': 'chattanooga',
    'Chicago St.': 'chicago-state',
    'Cincinnati': 'cincinnati',
    'Citadel': 'citadel',
    'Clemson': 'clemson',
    'Cleveland St.': 'cleveland-state',
    'Coastal Carolina': 'coastal-carolina',
    'Colgate': 'colgate',
    'Colorado': 'colorado',
    'Colorado St.': 'colorado-state',
    'Columbia': 'columbia',
    'Connecticut': 'connecticut',
    'UConn': 'connecticut',
    'Coppin St.': 'coppin-state',
    'Cornell': 'cornell',
    'Creighton': 'creighton',
    'Dartmouth': 'dartmouth',
    'Davidson': 'davidson',
    'Dayton': 'dayton',
    'DePaul': 'depaul',
    'Delaware': 'delaware',
    'Delaware St.': 'delaware-state',
    'Denver': 'denver',
    'Detroit': 'detroit-mercy',
    'Drake': 'drake',
    'Drexel': 'drexel',
    'Duke': 'duke',
    'Duquesne': 'duquesne',
    'East Carolina': 'east-carolina',
    'East Tennessee St.': 'east-tennessee-state',
    'Eastern Illinois': 'eastern-illinois',
    'Eastern Kentucky': 'eastern-kentucky',
    'Eastern Michigan': 'eastern-michigan',
    'Eastern Washington': 'eastern-washington',
    'Elon': 'elon',
    'Evansville': 'evansville',
    'Fairfield': 'fairfield',
    'Fairleigh Dickinson': 'fairleigh-dickinson',
    'Florida': 'florida',
    'Florida A&M': 'florida-am',
    'Florida Atlantic': 'florida-atlantic',
    'Florida Gulf Coast': 'florida-gulf-coast',
    'Florida International': 'florida-international',
    'Florida St.': 'florida-state',
    'Fordham': 'fordham',
    'Fresno St.': 'fresno-state',
    'Furman': 'furman',
    'Gardner Webb': 'gardner-webb',
    'George Mason': 'george-mason',
    'George Washington': 'george-washington',
    'Georgetown': 'georgetown',
    'Georgia': 'georgia',
    'Georgia Southern': 'georgia-southern',
    'Georgia St.': 'georgia-state',
    'Georgia Tech': 'georgia-tech',
    'Gonzaga': 'gonzaga',
    'Grambling': 'grambling',
    'Grand Canyon': 'grand-canyon',
    'Green Bay': 'green-bay',
    'Hampton': 'hampton',
    'Hartford': 'hartford',
    'Harvard': 'harvard',
    'Hawaii': 'hawaii',
    'High Point': 'high-point',
    'Hofstra': 'hofstra',
    'Holy Cross': 'holy-cross',
    'Houston': 'houston',
    'Houston Baptist': 'houston-baptist',
    'Howard': 'howard',
    'IPFW': 'ipfw',
    'Idaho': 'idaho',
    'Idaho St.': 'idaho-state',
    'Illinois': 'illinois',
    'Illinois Chicago': 'illinois-chicago',
    'Illinois St.': 'illinois-state',
    'Incarnate Word': 'incarnate-word',
    'Indiana': 'indiana',
    'Indiana St.': 'indiana-state',
    'Iona': 'iona',
    'Iowa': 'iowa',
    'Iowa St.': 'iowa-state',
    'IUPUI': 'iupui',
    'Jackson St.': 'jackson-state',
    'Jacksonville': 'jacksonville',
    'Jacksonville St.': 'jacksonville-state',
    'James Madison': 'james-madison',
    'Kansas': 'kansas',
    'Kansas St.': 'kansas-state',
    'Kennesaw St.': 'kennesaw-state',
    'Kent St.': 'kent-state',
    'Kentucky': 'kentucky',
    'LSU': 'louisiana-state',
    'La Salle': 'la-salle',
    'Lafayette': 'lafayette',
    'Lamar': 'lamar',
    'Lehigh': 'lehigh',
    'Liberty': 'liberty',
    'Lipscomb': 'lipscomb',
    'Long Beach St.': 'long-beach-state',
    'Long Island University': 'long-island-university',
    'Longwood': 'longwood',
    'Louisiana': 'louisiana-lafayette',
    'Louisiana Lafayette': 'louisiana-lafayette',
    'Louisiana Monroe': 'louisiana-monroe',
    'Louisiana Tech': 'louisiana-tech',
    'Louisville': 'louisville',
    'Loyola Chicago': 'loyola-il',
    'Loyola MD': 'loyola-md',
    'Loyola Marymount': 'loyola-marymount',
    'Maine': 'maine',
    'Manhattan': 'manhattan',
    'Marist': 'marist',
    'Marquette': 'marquette',
    'Marshall': 'marshall',
    'Maryland': 'maryland',
    'Maryland Eastern Shore': 'maryland-eastern-shore',
    'Massachusetts': 'massachusetts',
    'McNeese St.': 'mcneese-state',
    'Memphis': 'memphis',
    'Mercer': 'mercer',
    'Miami FL': 'miami-fl',
    'Miami OH': 'miami-oh',
    'Michigan': 'michigan',
    'Michigan St.': 'michigan-state',
    'Middle Tennessee': 'middle-tennessee',
    'Milwaukee': 'milwaukee',
    'Minnesota': 'minnesota',
    'Mississippi': 'mississippi',
    'Ole Miss': 'mississippi',
    'Mississippi St.': 'mississippi-state',
    'Mississippi Valley St.': 'mississippi-valley-state',
    'Missouri': 'missouri',
    'Missouri Kansas City': 'missouri-kansas-city',
    'Missouri St.': 'missouri-state',
    'Monmouth': 'monmouth',
    'Montana': 'montana',
    'Montana St.': 'montana-state',
    'Morehead St.': 'morehead-state',
    'Morgan St.': 'morgan-state',
    'Mount St. Mary\'s': 'mount-st-marys',
    'Murray St.': 'murray-state',
    'NC State': 'north-carolina-state',
    'N.C. State': 'north-carolina-state',
    'Navy': 'navy',
    'Nebraska': 'nebraska',
    'Nevada': 'nevada',
    'New Hampshire': 'new-hampshire',
    'New Mexico': 'new-mexico',
    'New Mexico St.': 'new-mexico-state',
    'New Orleans': 'new-orleans',
    'Niagara': 'niagara',
    'Nicholls St.': 'nicholls-state',
    'Norfolk St.': 'norfolk-state',
    'North Carolina': 'north-carolina',
    'North Carolina A&T': 'north-carolina-at',
    'North Carolina Central': 'north-carolina-central',
    'North Dakota': 'north-dakota',
    'North Dakota St.': 'north-dakota-state',
    'North Florida': 'north-florida',
    'North Texas': 'north-texas',
    'Northeastern': 'northeastern',
    'Northern Arizona': 'northern-arizona',
    'Northern Colorado': 'northern-colorado',
    'Northern Illinois': 'northern-illinois',
    'Northern Iowa': 'northern-iowa',
    'Northern Kentucky': 'northern-kentucky',
    'Northwestern': 'northwestern',
    'Northwestern St.': 'northwestern-state',
    'Notre Dame': 'notre-dame',
    'Oakland': 'oakland',
    'Ohio': 'ohio',
    'Ohio St.': 'ohio-state',
    'Oklahoma': 'oklahoma',
    'Oklahoma St.': 'oklahoma-state',
    'Old Dominion': 'old-dominion',
    'Oral Roberts': 'oral-roberts',
    'Oregon': 'oregon',
    'Oregon St.': 'oregon-state',
    'Pacific': 'pacific',
    'Penn': 'pennsylvania',
    'Penn St.': 'penn-state',
    'Pepperdine': 'pepperdine',
    'Pittsburgh': 'pittsburgh',
    'Pitt': 'pittsburgh',
    'Portland': 'portland',
    'Portland St.': 'portland-state',
    'Prairie View A&M': 'prairie-view',
    'Presbyterian': 'presbyterian',
    'Princeton': 'princeton',
    'Providence': 'providence',
    'Purdue': 'purdue',
    'Quinnipiac': 'quinnipiac',
    'Radford': 'radford',
    'Rhode Island': 'rhode-island',
    'Rice': 'rice',
    'Richmond': 'richmond',
    'Rider': 'rider',
    'Robert Morris': 'robert-morris',
    'Rutgers': 'rutgers',
    'SIU Edwardsville': 'southern-illinois-edwardsville',
    'SMU': 'southern-methodist',
    'Sacramento St.': 'sacramento-state',
    'Sacred Heart': 'sacred-heart',
    'Sam Houston St.': 'sam-houston-state',
    'Samford': 'samford',
    'San Diego': 'san-diego',
    'San Diego St.': 'san-diego-state',
    'San Francisco': 'san-francisco',
    'San Jose St.': 'san-jose-state',
    'Santa Clara': 'santa-clara',
    'Savannah St.': 'savannah-state',
    'Seattle': 'seattle',
    'Seton Hall': 'seton-hall',
    'Siena': 'siena',
    'South Alabama': 'south-alabama',
    'South Carolina': 'south-carolina',
    'South Carolina St.': 'south-carolina-state',
    'South Dakota': 'south-dakota',
    'South Dakota St.': 'south-dakota-state',
    'South Florida': 'south-florida',
    'Southeast Missouri St.': 'southeast-missouri-state',
    'Southeastern Louisiana': 'southeastern-louisiana',
    'Southern': 'southern',
    'Southern Illinois': 'southern-illinois',
    'Southern Miss': 'southern-mississippi',
    'Southern Utah': 'southern-utah',
    'St. Bonaventure': 'st-bonaventure',
    "St. John's": 'st-johns-ny',
    "Saint John's": 'st-johns-ny',
    "Saint Joseph's": 'saint-josephs',
    "St. Joseph's": 'saint-josephs',
    "Saint Louis": 'saint-louis',
    "Saint Mary's": 'saint-marys-ca',
    "Saint Peter's": 'saint-peters',
    "St. Peter's": 'saint-peters',
    'Stanford': 'stanford',
    'Stetson': 'stetson',
    'Stony Brook': 'stony-brook',
    'Syracuse': 'syracuse',
    'TCU': 'texas-christian',
    'Temple': 'temple',
    'Tennessee': 'tennessee',
    'Tennessee Martin': 'tennessee-martin',
    'Tennessee St.': 'tennessee-state',
    'Tennessee Tech': 'tennessee-tech',
    'Texas': 'texas',
    'Texas A&M': 'texas-am',
    'Texas A&M Corpus Christi': 'texas-am-corpus-christi',
    'Texas Arlington': 'texas-arlington',
    'Texas Pan American': 'texas-pan-american',
    'Texas San Antonio': 'texas-san-antonio',
    'Texas Southern': 'texas-southern',
    'Texas St.': 'texas-state',
    'Texas Tech': 'texas-tech',
    'The Citadel': 'citadel',
    'Toledo': 'toledo',
    'Towson': 'towson',
    'Troy': 'troy',
    'Tulane': 'tulane',
    'Tulsa': 'tulsa',
    'UAB': 'alabama-birmingham',
    'UC Davis': 'uc-davis',
    'UC Irvine': 'uc-irvine',
    'UC Riverside': 'uc-riverside',
    'UC Santa Barbara': 'uc-santa-barbara',
    'UCF': 'central-florida',
    'UCLA': 'ucla',
    'UMBC': 'maryland-baltimore-county',
    'UMKC': 'missouri-kansas-city',
    'UMass Lowell': 'massachusetts-lowell',
    'UNCG': 'north-carolina-greensboro',
    'UNC Asheville': 'north-carolina-asheville',
    'UNC Greensboro': 'north-carolina-greensboro',
    'UNC Wilmington': 'north-carolina-wilmington',
    'UNLV': 'nevada-las-vegas',
    'USC': 'southern-california',
    'USC Upstate': 'south-carolina-upstate',
    'UT Rio Grande Valley': 'texas-rio-grande-valley',
    'UTEP': 'texas-el-paso',
    'UTSA': 'texas-san-antonio',
    'Utah': 'utah',
    'Utah St.': 'utah-state',
    'Utah Valley': 'utah-valley',
    'VCU': 'virginia-commonwealth',
    'VMI': 'virginia-military-institute',
    'Valparaiso': 'valparaiso',
    'Vanderbilt': 'vanderbilt',
    'Vermont': 'vermont',
    'Villanova': 'villanova',
    'Virginia': 'virginia',
    'Virginia Tech': 'virginia-tech',
    'Wagner': 'wagner',
    'Wake Forest': 'wake-forest',
    'Washington': 'washington',
    'Washington St.': 'washington-state',
    'Weber St.': 'weber-state',
    'West Virginia': 'west-virginia',
    'Western Carolina': 'western-carolina',
    'Western Illinois': 'western-illinois',
    'Western Kentucky': 'western-kentucky',
    'Western Michigan': 'western-michigan',
    'Wichita St.': 'wichita-state',
    'William & Mary': 'william-mary',
    'Winthrop': 'winthrop',
    'Wisconsin': 'wisconsin',
    'Wofford': 'wofford',
    'Wright St.': 'wright-state',
    'Wyoming': 'wyoming',
    'Xavier': 'xavier',
    'Yale': 'yale',
    'Youngstown St.': 'youngstown-state',
}


def get_slug(team_name):
    """Get Sports-Reference slug for a team name."""
    if team_name in EXTENDED_TEAM_SLUGS:
        return EXTENDED_TEAM_SLUGS[team_name]
    slug = team_name.lower()
    slug = re.sub(r"[^a-z0-9\s\-]", "", slug)
    slug = slug.strip().replace(" ", "-")
    return slug


CLASS_VALUES = {
    'fr': 0, 'freshman': 0,
    'so': 1, 'sophomore': 1,
    'jr': 2, 'junior': 2,
    'sr': 3, 'senior': 3,
    'gr': 3, 'graduate': 3,
    'rs fr': 0, 'rs so': 1, 'rs jr': 2, 'rs sr': 3,
}


def parse_height_inches(h):
    """Convert height string to inches."""
    if pd.isna(h) or not h:
        return None
    h = str(h).strip()
    m = re.match(r"(\d+)-(\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    m = re.match(r"(\d+)'(\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    try:
        v = float(h)
        return v if v > 12 else v * 12
    except ValueError:
        return None


def scrape_sr_team(session, team_name, year, delay=3.0):
    """
    Scrape Sports-Reference for a team's roster and stats in a single request.
    Returns dict with EXP, AVG_HGT, EFF_HGT, or None on failure.
    """
    slug = get_slug(team_name)
    url = f"https://www.sports-reference.com/cbb/schools/{slug}/{year}.html"

    time.sleep(delay)

    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 429:
            logger.warning(f"Rate limited on {team_name} {year}, waiting 30s...")
            time.sleep(30)
            resp = session.get(url, timeout=15)

        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Parse roster table for class and height
        roster_table = soup.find("table", {"id": "roster"})
        roster_info = {}
        if roster_table:
            for row in roster_table.find_all("tr")[1:]:
                cols = row.find_all(["td", "th"])
                if len(cols) < 5:
                    continue
                name = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                player_class = cols[2].get_text(strip=True).lower() if len(cols) > 2 else ""
                height = cols[4].get_text(strip=True) if len(cols) > 4 else ""
                if name:
                    roster_info[name.lower()] = {
                        "class": player_class,
                        "height": parse_height_inches(height),
                    }

        # Parse per-game stats for minutes
        # SR sometimes wraps tables in comments for deferred rendering
        per_game = soup.find("table", {"id": "per_game"})
        if not per_game:
            comments = soup.find_all(string=lambda t: isinstance(t, str) and 'id="per_game"' in t)
            for c in comments:
                csoup = BeautifulSoup(c, "html.parser")
                per_game = csoup.find("table", {"id": "per_game"})
                if per_game:
                    break

        minutes_map = {}
        if per_game:
            for row in per_game.find_all("tr"):
                name_td = row.find("td", {"data-stat": "player"})
                mp_td = row.find("td", {"data-stat": "mp_per_g"})
                g_td = row.find("td", {"data-stat": "g"})
                if name_td and mp_td and g_td:
                    try:
                        name = name_td.get_text(strip=True).lower()
                        mpg = float(mp_td.get_text(strip=True) or 0)
                        games = int(g_td.get_text(strip=True) or 0)
                        minutes_map[name] = mpg * games
                    except (ValueError, TypeError):
                        pass

        if not roster_info:
            return None

        # Merge and calculate
        players = []
        for name, info in roster_info.items():
            mins = minutes_map.get(name, 0)
            if mins <= 0:
                for mname, mval in minutes_map.items():
                    if name.split()[-1] == mname.split()[-1] and name[0] == mname[0]:
                        mins = mval
                        break

            class_val = CLASS_VALUES.get(info["class"], 1.5)
            players.append({
                "class_val": class_val,
                "height": info["height"],
                "minutes": mins,
            })

        players_with_mins = [p for p in players if p["minutes"] > 0]
        if not players_with_mins:
            players_with_mins = players
            for p in players_with_mins:
                p["minutes"] = 100

        total_mins = sum(p["minutes"] for p in players_with_mins)
        if total_mins == 0:
            return None

        # EXP: minutes-weighted class average
        exp_val = sum(p["class_val"] * p["minutes"] for p in players_with_mins) / total_mins

        # AVG HGT: minutes-weighted average height
        hgt_players = [p for p in players_with_mins if p["height"] is not None]
        if hgt_players:
            hgt_mins = sum(p["minutes"] for p in hgt_players)
            avg_hgt = sum(p["height"] * p["minutes"] for p in hgt_players) / hgt_mins if hgt_mins > 0 else None

            # EFF HGT: avg height of tallest players covering 40% of minutes
            hgt_players_sorted = sorted(hgt_players, key=lambda x: x["height"], reverse=True)
            target = total_mins * 0.40
            acc = 0
            interior = []
            for p in hgt_players_sorted:
                interior.append(p)
                acc += p["minutes"]
                if acc >= target:
                    break
            int_mins = sum(p["minutes"] for p in interior)
            eff_hgt = sum(p["height"] * p["minutes"] for p in interior) / int_mins if int_mins > 0 else None
        else:
            avg_hgt = None
            eff_hgt = None

        return {
            "EXP": round(exp_val, 3),
            "AVG HGT": round(avg_hgt, 1) if avg_hgt else None,
            "EFF HGT": round(eff_hgt, 1) if eff_hgt else None,
            "roster_size": len(roster_info),
            "players_with_mins": len(players_with_mins),
        }

    except Exception as e:
        logger.error(f"Error scraping {team_name} {year}: {e}")
        return None


def fetch_rosters_for_years(years=None):
    """Scrape Sports-Reference for EXP/HGT data for specified years."""
    import requests as req

    if years is None:
        years = [2002, 2003, 2004, 2005, 2006]

    main_df = pd.read_csv(RAW_DIR / "KenPom Barttorvik.csv")

    session = req.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    })

    print("\n" + "=" * 70)
    print(f"PHASE 3: Sports-Reference Rosters ({years})")
    print("=" * 70)

    all_results = []

    for year in years:
        year_teams = main_df[main_df["YEAR"] == year]["TEAM"].tolist()
        existing_file = ROSTER_DIR / f"roster_{year}.csv"

        already_done = set()
        if existing_file.exists():
            done_df = pd.read_csv(existing_file)
            already_done = set(done_df["TEAM"].tolist())
            all_results.extend(done_df.to_dict("records"))
            print(f"\n  {year}: {len(already_done)} teams already scraped, {len(year_teams) - len(already_done)} remaining")

        remaining = [t for t in year_teams if t not in already_done]
        total = len(remaining)

        if total == 0:
            print(f"  {year}: All teams already scraped")
            continue

        print(f"\n  {year}: Scraping {total} teams (3s delay each, ~{total*3//60}m{total*3%60}s estimated)")

        year_results = []
        success = 0
        fail = 0

        for i, team in enumerate(remaining):
            if (i + 1) % 25 == 0 or i == 0:
                print(f"    [{i+1}/{total}] {team}...", end=" ", flush=True)

            result = scrape_sr_team(session, team, year, delay=3.0)

            if result:
                record = {"TEAM": team, "YEAR": year, **result}
                year_results.append(record)
                success += 1
                if (i + 1) % 25 == 0 or i == 0:
                    print(f"EXP={result['EXP']:.2f}, HGT={result.get('AVG HGT', 'N/A')}")
            else:
                year_results.append({"TEAM": team, "YEAR": year, "EXP": None, "AVG HGT": None, "EFF HGT": None})
                fail += 1
                if (i + 1) % 25 == 0 or i == 0:
                    print("FAILED")

            # Save progress every 50 teams
            if (i + 1) % 50 == 0:
                progress_df = pd.DataFrame(list(already_done_records) + year_results) if already_done else pd.DataFrame(year_results)
                if already_done:
                    existing_records = pd.read_csv(existing_file).to_dict("records") if existing_file.exists() else []
                    progress_df = pd.DataFrame(existing_records + year_results)
                else:
                    progress_df = pd.DataFrame(year_results)
                progress_df.to_csv(existing_file, index=False)

        # Save year results
        if already_done:
            existing_records = pd.read_csv(existing_file).to_dict("records")
            all_year = existing_records + year_results
        else:
            all_year = year_results

        pd.DataFrame(all_year).to_csv(existing_file, index=False)
        all_results.extend(year_results)

        print(f"  {year}: {success} succeeded, {fail} failed out of {total}")

    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(ROSTER_DIR / "roster_all.csv", index=False)
        print(f"\nSaved {len(df_all)} total roster records")

    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


# ============================================================================
# INTEGRATION: Merge collected data into main dataset
# ============================================================================

def integrate_all():
    """Merge all collected data into the main KenPom Barttorvik.csv."""
    main_file = RAW_DIR / "KenPom Barttorvik.csv"
    df = pd.read_csv(main_file)
    print("\n" + "=" * 70)
    print("INTEGRATING COLLECTED DATA")
    print("=" * 70)
    print(f"Main dataset: {len(df)} rows, {len(df.columns)} columns")

    changes = {"elite_sos": 0, "talent": 0, "exp": 0, "hgt": 0}

    # --- Elite SOS ---
    esos_file = BART_DIR / "elite_sos_all.csv"
    if esos_file.exists():
        esos_df = pd.read_csv(esos_file)
        print(f"\nElite SOS data: {len(esos_df)} records")

        if "ELITE SOS" not in df.columns:
            df["ELITE SOS"] = np.nan

        for _, row in esos_df.iterrows():
            mask = (df["TEAM"] == row["TEAM"]) & (df["YEAR"] == row["YEAR"])
            if mask.any() and pd.isna(df.loc[mask, "ELITE SOS"]).all():
                df.loc[mask, "ELITE SOS"] = row["ELITE SOS"]
                changes["elite_sos"] += mask.sum()

        # Fuzzy matching for team names that differ between KenPom and Barttorvik
        unmatched = df[df["ELITE SOS"].isna()]["YEAR"].value_counts()
        if not unmatched.empty:
            print(f"  Direct match: {changes['elite_sos']} filled")
            esos_teams = set(esos_df["TEAM"].unique())
            kp_teams = set(df["TEAM"].unique())
            only_bart = esos_teams - kp_teams
            only_kp = kp_teams - esos_teams
            if only_bart and only_kp:
                name_map = fuzzy_team_match(only_bart, only_kp)
                for bart_name, kp_name in name_map.items():
                    esos_rows = esos_df[esos_df["TEAM"] == bart_name]
                    for _, row in esos_rows.iterrows():
                        mask = (df["TEAM"] == kp_name) & (df["YEAR"] == row["YEAR"])
                        if mask.any() and pd.isna(df.loc[mask, "ELITE SOS"]).all():
                            df.loc[mask, "ELITE SOS"] = row["ELITE SOS"]
                            changes["elite_sos"] += mask.sum()
                print(f"  After fuzzy match: {changes['elite_sos']} total filled")
    else:
        print("\nNo Elite SOS data found")

    # --- Talent ---
    talent_file = TALENT_DIR / "talent_all.csv"
    if talent_file.exists():
        talent_df = pd.read_csv(talent_file)
        print(f"\nTalent data: {len(talent_df)} records")

        if "TALENT" not in df.columns:
            df["TALENT"] = np.nan

        for _, row in talent_df.iterrows():
            if pd.isna(row.get("TALENT_SCORE")):
                continue
            mask = (df["TEAM"] == row["TEAM"]) & (df["YEAR"] == row["YEAR"])
            if mask.any() and pd.isna(df.loc[mask, "TALENT"]).all():
                df.loc[mask, "TALENT"] = row["TALENT_SCORE"]
                changes["talent"] += mask.sum()

        if changes["talent"] > 0:
            print(f"  Direct match: {changes['talent']} filled")

        # Fuzzy match
        talent_teams = set(talent_df["TEAM"].unique())
        kp_teams = set(df["TEAM"].unique())
        only_247 = talent_teams - kp_teams
        only_kp = kp_teams - talent_teams
        if only_247 and only_kp:
            name_map = fuzzy_team_match(only_247, only_kp)
            for t247_name, kp_name in name_map.items():
                t_rows = talent_df[talent_df["TEAM"] == t247_name]
                for _, row in t_rows.iterrows():
                    if pd.isna(row.get("TALENT_SCORE")):
                        continue
                    mask = (df["TEAM"] == kp_name) & (df["YEAR"] == row["YEAR"])
                    if mask.any() and pd.isna(df.loc[mask, "TALENT"]).all():
                        df.loc[mask, "TALENT"] = row["TALENT_SCORE"]
                        changes["talent"] += mask.sum()
            if changes["talent"] > 0:
                print(f"  After fuzzy match: {changes['talent']} total filled")
    else:
        print("\nNo Talent data found")

    # --- EXP/HGT from rosters ---
    roster_file = ROSTER_DIR / "roster_all.csv"
    if roster_file.exists():
        roster_df = pd.read_csv(roster_file)
        print(f"\nRoster data: {len(roster_df)} records")

        for _, row in roster_df.iterrows():
            mask = (df["TEAM"] == row["TEAM"]) & (df["YEAR"] == row["YEAR"])
            if not mask.any():
                continue

            if pd.notna(row.get("EXP")) and pd.isna(df.loc[mask, "EXP"]).all():
                df.loc[mask, "EXP"] = row["EXP"]
                changes["exp"] += mask.sum()

            if pd.notna(row.get("AVG HGT")) and pd.isna(df.loc[mask, "AVG HGT"]).all():
                df.loc[mask, "AVG HGT"] = row["AVG HGT"]
                changes["hgt"] += mask.sum()

            if pd.notna(row.get("EFF HGT")) and pd.isna(df.loc[mask, "EFF HGT"]).all():
                df.loc[mask, "EFF HGT"] = row["EFF HGT"]

        print(f"  EXP filled: {changes['exp']}")
        print(f"  HGT filled: {changes['hgt']}")
    else:
        print("\nNo roster data found")

    # Save updated dataset
    backup = RAW_DIR / "KenPom Barttorvik_backup.csv"
    pd.read_csv(main_file).to_csv(backup, index=False)
    print(f"\nBackup saved to {backup}")

    df.to_csv(main_file, index=False)
    print(f"Updated dataset saved: {len(df)} rows, {len(df.columns)} columns")

    # Coverage report
    print("\n" + "-" * 50)
    print("UPDATED COVERAGE REPORT")
    print("-" * 50)
    for col in ["ELITE SOS", "TALENT", "EXP", "AVG HGT", "EFF HGT"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            pct = valid / len(df) * 100
            print(f"  {col:12}: {valid:5}/{len(df)} ({pct:.1f}%)")

            if valid < len(df):
                missing_years = sorted(df[df[col].isna()]["YEAR"].unique())
                if len(missing_years) <= 8:
                    print(f"               Missing in: {missing_years}")
        else:
            print(f"  {col:12}: NOT IN DATA")

    return df


def fuzzy_team_match(source_names, target_names):
    """Build a name mapping between two sets using simple heuristics."""
    mapping = {}
    target_list = list(target_names)

    abbreviation_map = {
        "St.": "State", "State": "St.",
        "Univ": "University", "University": "Univ",
    }

    for src in source_names:
        src_lower = src.lower().strip()
        src_words = set(src_lower.replace(".", "").replace("'", "").split())

        best_match = None
        best_score = 0

        for tgt in target_list:
            tgt_lower = tgt.lower().strip()
            tgt_words = set(tgt_lower.replace(".", "").replace("'", "").split())

            common = src_words & tgt_words
            total = src_words | tgt_words
            score = len(common) / max(len(total), 1)

            if score > best_score and score > 0.5:
                best_score = score
                best_match = tgt

        if best_match:
            mapping[src] = best_match

    return mapping


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect missing NCAA data")
    parser.add_argument("--phase1", action="store_true", help="Barttorvik Elite SOS")
    parser.add_argument("--phase2", action="store_true", help="247Sports Talent")
    parser.add_argument("--phase3", action="store_true", help="Sports-Reference EXP/HGT")
    parser.add_argument("--integrate", action="store_true", help="Merge all into main dataset")
    parser.add_argument("--all", action="store_true", help="Run all phases + integrate")
    parser.add_argument("--years", type=str, default=None, help="Years for phase3 (e.g. '2002-2006')")

    args = parser.parse_args()

    if not any([args.phase1, args.phase2, args.phase3, args.integrate, args.all]):
        parser.print_help()
        print("\n\nStatus of collected data:")
        for label, path in [
            ("Elite SOS", BART_DIR / "elite_sos_all.csv"),
            ("Talent", TALENT_DIR / "talent_all.csv"),
            ("Recruits", TALENT_DIR / "recruits_all.csv"),
            ("Rosters", ROSTER_DIR / "roster_all.csv"),
        ]:
            if path.exists():
                df = pd.read_csv(path)
                years = sorted(df["YEAR"].unique()) if "YEAR" in df.columns else sorted(df["year"].unique()) if "year" in df.columns else []
                print(f"  {label:12}: {len(df)} records, years {years[0]}-{years[-1]}" if years else f"  {label:12}: {len(df)} records")
            else:
                print(f"  {label:12}: Not yet collected")
        return

    if args.phase1 or args.all:
        asyncio.run(fetch_elite_sos_all_years())

    if args.phase2 or args.all:
        asyncio.run(fetch_talent_all_years())

    if args.phase3 or args.all:
        years = None
        if args.years:
            start, end = map(int, args.years.split("-"))
            years = list(range(start, end + 1))
        fetch_rosters_for_years(years)

    if args.integrate or args.all:
        integrate_all()


if __name__ == "__main__":
    main()
