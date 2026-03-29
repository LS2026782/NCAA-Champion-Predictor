"""
Sports-Reference roster scraper for EXP and HGT (2002-2006).

Smart strategy:
  Phase A — Scrape ~365 TOURNAMENT teams first (mandatory for model training)
  Phase B — Scrape remaining ~1,279 non-tournament teams (optional, fills gaps)

Single Playwright worker with 6s delay respects SR's rate limits.
Phase A takes ~35 min, Phase B takes ~2h (run overnight or skip entirely).
"""
import asyncio
import re
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ROSTER_DIR = DATA_DIR / "historical" / "rosters"
ROSTER_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

TEAM_SLUGS = {
    'Air Force': 'air-force', 'Akron': 'akron', 'Alabama': 'alabama',
    'Alabama A&M': 'alabama-am', 'Alabama St.': 'alabama-state',
    'Albany': 'albany-ny', 'Alcorn St.': 'alcorn-state',
    'American': 'american', 'Appalachian St.': 'appalachian-state',
    'Arizona': 'arizona', 'Arizona St.': 'arizona-state',
    'Arkansas': 'arkansas', 'Arkansas Little Rock': 'arkansas-little-rock',
    'Arkansas Pine Bluff': 'arkansas-pine-bluff', 'Arkansas St.': 'arkansas-state',
    'Army': 'army', 'Auburn': 'auburn', 'Austin Peay': 'austin-peay',
    'BYU': 'brigham-young', 'Ball St.': 'ball-state', 'Baylor': 'baylor',
    'Belmont': 'belmont', 'Bethune Cookman': 'bethune-cookman',
    'Binghamton': 'binghamton', 'Birmingham Southern': 'birmingham-southern',
    'Boise St.': 'boise-state', 'Boston College': 'boston-college',
    'Boston University': 'boston-university', 'Bowling Green': 'bowling-green-state',
    'Bradley': 'bradley', 'Brown': 'brown', 'Bucknell': 'bucknell',
    'Buffalo': 'buffalo', 'Butler': 'butler', 'Cal Poly': 'cal-poly',
    'Cal St. Fullerton': 'cal-state-fullerton', 'Cal St. Northridge': 'cal-state-northridge',
    'California': 'california', 'Campbell': 'campbell', 'Canisius': 'canisius',
    'Centenary': 'centenary-la',
    'Central Connecticut': 'central-connecticut-state', 'Central Michigan': 'central-michigan',
    'Charleston': 'college-of-charleston', 'Charleston Southern': 'charleston-southern',
    'Charlotte': 'charlotte', 'Chattanooga': 'chattanooga',
    'Chicago St.': 'chicago-state', 'Cincinnati': 'cincinnati',
    'Citadel': 'citadel', 'The Citadel': 'citadel',
    'Clemson': 'clemson', 'Cleveland St.': 'cleveland-state',
    'Coastal Carolina': 'coastal-carolina', 'Colgate': 'colgate',
    'Colorado': 'colorado', 'Colorado St.': 'colorado-state',
    'Columbia': 'columbia', 'Connecticut': 'connecticut', 'UConn': 'connecticut',
    'Coppin St.': 'coppin-state', 'Cornell': 'cornell', 'Creighton': 'creighton',
    'Dartmouth': 'dartmouth', 'Davidson': 'davidson', 'Dayton': 'dayton',
    'DePaul': 'depaul', 'Delaware': 'delaware', 'Delaware St.': 'delaware-state',
    'Denver': 'denver', 'Detroit': 'detroit-mercy', 'Drake': 'drake',
    'Drexel': 'drexel', 'Duke': 'duke', 'Duquesne': 'duquesne',
    'East Carolina': 'east-carolina', 'East Tennessee St.': 'east-tennessee-state',
    'Eastern Illinois': 'eastern-illinois', 'Eastern Kentucky': 'eastern-kentucky',
    'Eastern Michigan': 'eastern-michigan', 'Eastern Washington': 'eastern-washington',
    'Elon': 'elon', 'Evansville': 'evansville',
    'Fairfield': 'fairfield', 'Fairleigh Dickinson': 'fairleigh-dickinson',
    'Florida': 'florida', 'Florida A&M': 'florida-am',
    'Florida Atlantic': 'florida-atlantic', 'Florida International': 'florida-international',
    'Florida St.': 'florida-state', 'Fordham': 'fordham',
    'Fresno St.': 'fresno-state', 'Furman': 'furman',
    'Gardner Webb': 'gardner-webb', 'George Mason': 'george-mason',
    'George Washington': 'george-washington', 'Georgetown': 'georgetown',
    'Georgia': 'georgia', 'Georgia Southern': 'georgia-southern',
    'Georgia St.': 'georgia-state', 'Georgia Tech': 'georgia-tech',
    'Gonzaga': 'gonzaga', 'Grambling': 'grambling',
    'Green Bay': 'green-bay', 'Hampton': 'hampton',
    'Hartford': 'hartford', 'Harvard': 'harvard', 'Hawaii': 'hawaii',
    'High Point': 'high-point', 'Hofstra': 'hofstra', 'Holy Cross': 'holy-cross',
    'Houston': 'houston', 'Howard': 'howard',
    'IPFW': 'ipfw', 'Idaho': 'idaho', 'Idaho St.': 'idaho-state',
    'Illinois': 'illinois', 'Illinois Chicago': 'illinois-chicago',
    'Illinois St.': 'illinois-state', 'Indiana': 'indiana',
    'Indiana St.': 'indiana-state', 'Iona': 'iona',
    'Iowa': 'iowa', 'Iowa St.': 'iowa-state', 'IUPUI': 'iupui',
    'Jackson St.': 'jackson-state', 'Jacksonville': 'jacksonville',
    'Jacksonville St.': 'jacksonville-state', 'James Madison': 'james-madison',
    'Kansas': 'kansas', 'Kansas St.': 'kansas-state',
    'Kennesaw St.': 'kennesaw-state', 'Kent St.': 'kent-state',
    'Kentucky': 'kentucky', 'LSU': 'louisiana-state',
    'La Salle': 'la-salle', 'Lafayette': 'lafayette', 'Lamar': 'lamar',
    'Lehigh': 'lehigh', 'Liberty': 'liberty', 'Lipscomb': 'lipscomb',
    'Long Beach St.': 'long-beach-state', 'Long Island University': 'long-island-university',
    'Longwood': 'longwood', 'Louisiana Lafayette': 'louisiana-lafayette',
    'Louisiana': 'louisiana-lafayette',
    'Louisiana Monroe': 'louisiana-monroe', 'Louisiana Tech': 'louisiana-tech',
    'Louisville': 'louisville', 'Loyola Chicago': 'loyola-il',
    'Loyola MD': 'loyola-md', 'Loyola Marymount': 'loyola-marymount',
    'Maine': 'maine', 'Manhattan': 'manhattan', 'Marist': 'marist',
    'Marquette': 'marquette', 'Marshall': 'marshall', 'Maryland': 'maryland',
    'Maryland Eastern Shore': 'maryland-eastern-shore',
    'Massachusetts': 'massachusetts', 'McNeese St.': 'mcneese-state',
    'Memphis': 'memphis', 'Mercer': 'mercer',
    'Miami FL': 'miami-fl', 'Miami OH': 'miami-oh',
    'Michigan': 'michigan', 'Michigan St.': 'michigan-state',
    'Middle Tennessee': 'middle-tennessee', 'Milwaukee': 'milwaukee',
    'Minnesota': 'minnesota', 'Mississippi': 'mississippi', 'Ole Miss': 'mississippi',
    'Mississippi St.': 'mississippi-state',
    'Mississippi Valley St.': 'mississippi-valley-state',
    'Missouri': 'missouri', 'Missouri Kansas City': 'missouri-kansas-city',
    'Missouri St.': 'missouri-state', 'Monmouth': 'monmouth',
    'Montana': 'montana', 'Montana St.': 'montana-state',
    'Morehead St.': 'morehead-state', 'Morgan St.': 'morgan-state',
    "Mount St. Mary's": 'mount-st-marys', 'Murray St.': 'murray-state',
    'NC State': 'north-carolina-state', 'N.C. State': 'north-carolina-state',
    'Navy': 'navy', 'Nebraska': 'nebraska', 'Nevada': 'nevada',
    'New Hampshire': 'new-hampshire', 'New Mexico': 'new-mexico',
    'New Mexico St.': 'new-mexico-state', 'New Orleans': 'new-orleans',
    'Niagara': 'niagara', 'Nicholls St.': 'nicholls-state',
    'Norfolk St.': 'norfolk-state', 'North Carolina': 'north-carolina',
    'North Carolina A&T': 'north-carolina-at',
    'North Carolina Central': 'north-carolina-central',
    'North Dakota St.': 'north-dakota-state', 'North Texas': 'north-texas',
    'Northeastern': 'northeastern', 'Northern Arizona': 'northern-arizona',
    'Northern Colorado': 'northern-colorado', 'Northern Illinois': 'northern-illinois',
    'Northern Iowa': 'northern-iowa', 'Northwestern': 'northwestern',
    'Northwestern St.': 'northwestern-state', 'Notre Dame': 'notre-dame',
    'Oakland': 'oakland', 'Ohio': 'ohio', 'Ohio St.': 'ohio-state',
    'Oklahoma': 'oklahoma', 'Oklahoma St.': 'oklahoma-state',
    'Old Dominion': 'old-dominion', 'Oral Roberts': 'oral-roberts',
    'Oregon': 'oregon', 'Oregon St.': 'oregon-state',
    'Pacific': 'pacific', 'Penn': 'pennsylvania', 'Penn St.': 'penn-state',
    'Pepperdine': 'pepperdine', 'Pittsburgh': 'pittsburgh', 'Pitt': 'pittsburgh',
    'Portland': 'portland', 'Portland St.': 'portland-state',
    'Prairie View A&M': 'prairie-view', 'Princeton': 'princeton',
    'Providence': 'providence', 'Purdue': 'purdue',
    'Quinnipiac': 'quinnipiac', 'Radford': 'radford',
    'Rhode Island': 'rhode-island', 'Rice': 'rice', 'Richmond': 'richmond',
    'Rider': 'rider', 'Robert Morris': 'robert-morris', 'Rutgers': 'rutgers',
    'SIU Edwardsville': 'southern-illinois-edwardsville',
    'SMU': 'southern-methodist', 'Sacramento St.': 'sacramento-state',
    'Sacred Heart': 'sacred-heart', 'Sam Houston St.': 'sam-houston-state',
    'Samford': 'samford', 'San Diego': 'san-diego',
    'San Diego St.': 'san-diego-state', 'San Francisco': 'san-francisco',
    'San Jose St.': 'san-jose-state', 'Santa Clara': 'santa-clara',
    'Savannah St.': 'savannah-state', 'Seattle': 'seattle',
    'Seton Hall': 'seton-hall', 'Siena': 'siena',
    'South Alabama': 'south-alabama', 'South Carolina': 'south-carolina',
    'South Carolina St.': 'south-carolina-state',
    'South Florida': 'south-florida',
    'Southeast Missouri St.': 'southeast-missouri-state',
    'Southeastern Louisiana': 'southeastern-louisiana',
    'Southern': 'southern', 'Southern Illinois': 'southern-illinois',
    'Southern Miss': 'southern-mississippi', 'Southern Utah': 'southern-utah',
    'Southwest Missouri St.': 'missouri-state',
    "St. Bonaventure": 'st-bonaventure', "St. John's": 'st-johns-ny',
    "Saint John's": 'st-johns-ny', "Saint Joseph's": 'saint-josephs',
    "St. Joseph's": 'saint-josephs', "Saint Louis": 'saint-louis',
    "Saint Mary's": 'saint-marys-ca', "Saint Peter's": 'saint-peters',
    "St. Peter's": 'saint-peters', 'Stanford': 'stanford',
    'Stetson': 'stetson', 'Stony Brook': 'stony-brook',
    'Syracuse': 'syracuse', 'TCU': 'texas-christian',
    'Temple': 'temple', 'Tennessee': 'tennessee',
    'Tennessee Martin': 'tennessee-martin', 'Tennessee St.': 'tennessee-state',
    'Tennessee Tech': 'tennessee-tech', 'Texas': 'texas',
    'Texas A&M': 'texas-am', 'Texas A&M Corpus Christi': 'texas-am-corpus-christi',
    'Texas Arlington': 'texas-arlington', 'Texas Pan American': 'texas-pan-american',
    'Texas San Antonio': 'texas-san-antonio', 'Texas Southern': 'texas-southern',
    'Texas St.': 'texas-state', 'Texas Tech': 'texas-tech',
    'Toledo': 'toledo', 'Towson': 'towson', 'Troy': 'troy',
    'Tulane': 'tulane', 'Tulsa': 'tulsa',
    'UAB': 'alabama-birmingham', 'UC Davis': 'uc-davis',
    'UC Irvine': 'uc-irvine', 'UC Riverside': 'uc-riverside',
    'UC Santa Barbara': 'uc-santa-barbara',
    'UCF': 'central-florida', 'UCLA': 'ucla',
    'UMBC': 'maryland-baltimore-county', 'UMKC': 'missouri-kansas-city',
    'UNCG': 'north-carolina-greensboro', 'UNC Greensboro': 'north-carolina-greensboro',
    'UNC Asheville': 'north-carolina-asheville',
    'UNC Wilmington': 'north-carolina-wilmington',
    'UNLV': 'nevada-las-vegas', 'USC': 'southern-california',
    'UTEP': 'texas-el-paso', 'UTSA': 'texas-san-antonio',
    'Utah': 'utah', 'Utah St.': 'utah-state',
    'VCU': 'virginia-commonwealth', 'VMI': 'virginia-military-institute',
    'Valparaiso': 'valparaiso', 'Vanderbilt': 'vanderbilt',
    'Vermont': 'vermont', 'Villanova': 'villanova',
    'Virginia': 'virginia', 'Virginia Tech': 'virginia-tech',
    'Wagner': 'wagner', 'Wake Forest': 'wake-forest',
    'Washington': 'washington', 'Washington St.': 'washington-state',
    'Weber St.': 'weber-state', 'West Virginia': 'west-virginia',
    'Western Carolina': 'western-carolina', 'Western Illinois': 'western-illinois',
    'Western Kentucky': 'western-kentucky', 'Western Michigan': 'western-michigan',
    'Wichita St.': 'wichita-state', 'William & Mary': 'william-mary',
    'Winston Salem St.': 'winston-salem-state',
    'Winthrop': 'winthrop', 'Wisconsin': 'wisconsin', 'Wofford': 'wofford',
    'Wright St.': 'wright-state', 'Wyoming': 'wyoming',
    'Xavier': 'xavier', 'Yale': 'yale', 'Youngstown St.': 'youngstown-state',
}

CLASS_MAP = {
    'fr': 0, 'freshman': 0, 'so': 1, 'sophomore': 1,
    'jr': 2, 'junior': 2, 'sr': 3, 'senior': 3,
    'gr': 3, 'graduate': 3, 'rs fr': 0, 'rs so': 1, 'rs jr': 2, 'rs sr': 3,
}

JS_EXTRACT = """() => {
    const result = { roster: [], stats: [] };
    const rosterTable = document.getElementById('roster');
    if (rosterTable) {
        rosterTable.querySelectorAll('tbody tr').forEach(row => {
            const cells = row.querySelectorAll('td, th');
            if (cells.length >= 5) {
                result.roster.push({
                    name: (cells[1]?.innerText || '').trim().toLowerCase(),
                    cls:  (cells[2]?.innerText || '').trim().toLowerCase(),
                    ht:   (cells[4]?.innerText || '').trim(),
                });
            }
        });
    }
    let pg = document.getElementById('per_game');
    if (!pg) {
        const tw = document.createTreeWalker(document, NodeFilter.SHOW_COMMENT);
        let c;
        while (c = tw.nextNode()) {
            if (c.nodeValue.includes('id="per_game"')) {
                const d = document.createElement('div');
                d.innerHTML = c.nodeValue;
                pg = d.querySelector('#per_game');
                break;
            }
        }
    }
    if (pg) {
        pg.querySelectorAll('tbody tr').forEach(row => {
            const p = row.querySelector('td[data-stat="player"]');
            const m = row.querySelector('td[data-stat="mp_per_g"]');
            const g = row.querySelector('td[data-stat="g"]');
            if (p && m && g) {
                result.stats.push({
                    name: p.innerText.trim().toLowerCase(),
                    minutes: (parseFloat(m.innerText) || 0) * (parseInt(g.innerText) || 0),
                });
            }
        });
    }
    return result;
}"""


def get_slug(team_name):
    if team_name in TEAM_SLUGS:
        return TEAM_SLUGS[team_name]
    slug = team_name.lower()
    slug = re.sub(r"[^a-z0-9\s\-]", "", slug)
    return slug.strip().replace(" ", "-")


def parse_height(h):
    if not h:
        return None
    m = re.match(r"(\d+)-(\d+)", h)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return None


def process_page_data(data):
    if not data or not data.get("roster"):
        return None

    stats_map = {s["name"]: s["minutes"] for s in data.get("stats", [])}

    players = []
    for p in data["roster"]:
        mins = stats_map.get(p["name"], 0)
        if mins <= 0:
            last = p["name"].split()[-1] if p["name"] else ""
            for sname, smins in stats_map.items():
                if last and sname.endswith(last) and p["name"][:1] == sname[:1]:
                    mins = smins
                    break
        players.append({
            "class_val": CLASS_MAP.get(p["cls"], 1.5),
            "height": parse_height(p["ht"]),
            "minutes": mins,
        })

    with_mins = [p for p in players if p["minutes"] > 0] or players
    if not with_mins:
        return None
    if not any(p["minutes"] > 0 for p in players):
        for p in with_mins:
            p["minutes"] = 100

    total = sum(p["minutes"] for p in with_mins)
    if total == 0:
        return None

    exp = sum(p["class_val"] * p["minutes"] for p in with_mins) / total

    hgt_p = [p for p in with_mins if p["height"]]
    avg_hgt = eff_hgt = None
    if hgt_p:
        hm = sum(p["minutes"] for p in hgt_p)
        if hm > 0:
            avg_hgt = sum(p["height"] * p["minutes"] for p in hgt_p) / hm
        sorted_h = sorted(hgt_p, key=lambda x: x["height"], reverse=True)
        target = total * 0.40
        acc, interior = 0, []
        for p in sorted_h:
            interior.append(p)
            acc += p["minutes"]
            if acc >= target:
                break
        im = sum(p["minutes"] for p in interior)
        if im > 0:
            eff_hgt = sum(p["height"] * p["minutes"] for p in interior) / im

    return {
        "EXP": round(exp, 3),
        "AVG HGT": round(avg_hgt, 1) if avg_hgt else None,
        "EFF HGT": round(eff_hgt, 1) if eff_hgt else None,
    }


async def scrape_single(page, team, year, delay_ms=6000):
    """Scrape one team-year page."""
    slug = get_slug(team)
    url = f"https://www.sports-reference.com/cbb/schools/{slug}/{year}.html"
    await asyncio.sleep(delay_ms / 1000)

    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        if resp and resp.status == 429:
            print(f"  [429] {team} {year} — pausing 90s")
            await asyncio.sleep(90)
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        if resp and resp.status == 200:
            data = await page.evaluate(JS_EXTRACT)
            return process_page_data(data)
    except Exception:
        pass
    return None


async def main():
    from playwright.async_api import async_playwright
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-b", action="store_true",
                        help="Also scrape non-tournament teams (adds ~2h)")
    parser.add_argument("--delay", type=int, default=6000,
                        help="Delay between requests in ms (default: 6000)")
    args = parser.parse_args()

    main_df = pd.read_csv(RAW_DIR / "KenPom Barttorvik.csv")
    years = [2002, 2003, 2004, 2005, 2006]

    # Load already-done records (keyed by TEAM|YEAR)
    done_map = {}
    for year in years:
        f = ROSTER_DIR / f"roster_{year}.csv"
        if f.exists():
            existing = pd.read_csv(f)
            for _, row in existing.iterrows():
                key = f"{row['TEAM']}|{year}"
                done_map[key] = row.to_dict()

    # Build prioritised task list:
    # Phase A = tournament teams (ROUND not null)
    # Phase B = non-tournament teams
    tourney_mask = main_df["ROUND"].notna()

    phase_a, phase_b = [], []
    for year in years:
        yr_df = main_df[main_df["YEAR"] == year]
        for _, row in yr_df.iterrows():
            key = f"{row['TEAM']}|{year}"
            if key in done_map:
                continue
            if pd.notna(row.get("ROUND")):
                phase_a.append((row["TEAM"], year))
            else:
                phase_b.append((row["TEAM"], year))

    already_done = len(done_map)
    print("=" * 62)
    print("SPORTS-REFERENCE ROSTER SCRAPER  (priority-first)")
    print("=" * 62)
    print(f"Already done     : {already_done}")
    print(f"Phase A (tourney): {len(phase_a)} teams  (~{len(phase_a)*args.delay//60000}m)")
    print(f"Phase B (non-t.) : {len(phase_b)} teams  (~{len(phase_b)*args.delay//60000}m)")
    if not args.phase_b:
        print("  [--phase-b not set: Phase B will be skipped]")
    print()

    tasks = phase_a + (phase_b if args.phase_b else [])
    if not tasks:
        print("Nothing left to scrape.")
    else:
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

            start = asyncio.get_event_loop().time()
            for i, (team, year) in enumerate(tasks):
                # Print progress every 10 teams
                if i % 10 == 0:
                    elapsed = asyncio.get_event_loop().time() - start
                    rate = (i + 1) / elapsed if elapsed > 0 else 1
                    remaining = (len(tasks) - i) / rate
                    label = "A" if (team, year) in set(phase_a) else "B"
                    print(
                        f"[{i+1:4}/{len(tasks)}] [{label}] {team} ({year})"
                        f"  {elapsed/60:.1f}m elapsed  ~{remaining/60:.1f}m left"
                    )

                result = await scrape_single(page, team, year, args.delay)
                key = f"{team}|{year}"
                if result:
                    done_map[key] = {"TEAM": team, "YEAR": year, **result}
                else:
                    done_map[key] = {"TEAM": team, "YEAR": year,
                                     "EXP": None, "AVG HGT": None, "EFF HGT": None}

                # Save progress every 30 teams
                if (i + 1) % 30 == 0:
                    _save_progress(done_map, years)
                    print(f"  -- saved progress --")

            await browser.close()

    # Final save
    _save_progress(done_map, years)
    print("\nDone. Final coverage:")
    for year in years:
        f = ROSTER_DIR / f"roster_{year}.csv"
        if f.exists():
            df = pd.read_csv(f)
            valid = df["EXP"].notna().sum()
            print(f"  {year}: {len(df)} teams, {valid} with EXP ({valid/len(df)*100:.0f}%)")


# Set needed for label in progress printer
phase_a_set = set()


def _save_progress(done_map, years):
    """Write per-year CSV files and combined file."""
    for year in years:
        records = [v for v in done_map.values() if v.get("YEAR") == year]
        if records:
            pd.DataFrame(records).to_csv(ROSTER_DIR / f"roster_{year}.csv", index=False)

    all_dfs = [
        pd.read_csv(ROSTER_DIR / f"roster_{y}.csv")
        for y in years
        if (ROSTER_DIR / f"roster_{y}.csv").exists()
    ]
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged.to_csv(ROSTER_DIR / "roster_all.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
