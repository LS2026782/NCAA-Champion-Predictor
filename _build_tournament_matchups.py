"""
Build Tournament Matchups.csv from ESPN API.

Fetches all NCAA tournament game results for 2002-2025 via the ESPN
scoreboard API, then formats them for the game_loader.py pipeline.

Output format: YEAR, CURRENT ROUND, BY ROUND NO, TEAM, SEED, ROUND, SCORE
"""
import json
import time
import re
from pathlib import Path

import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).parent
OUTPUT = PROJECT_ROOT / "data" / "raw" / "Tournament Matchups.csv"

SKIP_YEAR = 2020

TOURNAMENT_DATES = {
    2002: ["20020314", "20020315", "20020316", "20020317", "20020321", "20020322", "20020323", "20020324", "20020330", "20020401"],
    2003: ["20030320", "20030321", "20030322", "20030323", "20030327", "20030328", "20030329", "20030330", "20030405", "20030407"],
    2004: ["20040318", "20040319", "20040320", "20040321", "20040325", "20040326", "20040327", "20040328", "20040403", "20040405"],
    2005: ["20050317", "20050318", "20050319", "20050320", "20050324", "20050325", "20050326", "20050327", "20050402", "20050404"],
    2006: ["20060316", "20060317", "20060318", "20060319", "20060323", "20060324", "20060325", "20060326", "20060401", "20060403"],
    2007: ["20070315", "20070316", "20070317", "20070318", "20070322", "20070323", "20070324", "20070325", "20070331", "20070402"],
    2008: ["20080320", "20080321", "20080322", "20080323", "20080327", "20080328", "20080329", "20080330", "20080405", "20080407"],
    2009: ["20090319", "20090320", "20090321", "20090322", "20090326", "20090327", "20090328", "20090329", "20090404", "20090406"],
    2010: ["20100318", "20100319", "20100320", "20100321", "20100325", "20100326", "20100327", "20100328", "20100403", "20100405"],
    2011: ["20110317", "20110318", "20110319", "20110320", "20110324", "20110325", "20110326", "20110327", "20110402", "20110404"],
    2012: ["20120315", "20120316", "20120317", "20120318", "20120322", "20120323", "20120324", "20120325", "20120331", "20120402"],
    2013: ["20130321", "20130322", "20130323", "20130324", "20130328", "20130329", "20130330", "20130331", "20130406", "20130408"],
    2014: ["20140320", "20140321", "20140322", "20140323", "20140327", "20140328", "20140329", "20140330", "20140405", "20140407"],
    2015: ["20150319", "20150320", "20150321", "20150322", "20150326", "20150327", "20150328", "20150329", "20150404", "20150406"],
    2016: ["20160317", "20160318", "20160319", "20160320", "20160324", "20160325", "20160326", "20160327", "20160402", "20160404"],
    2017: ["20170316", "20170317", "20170318", "20170319", "20170323", "20170324", "20170325", "20170326", "20170401", "20170403"],
    2018: ["20180315", "20180316", "20180317", "20180318", "20180322", "20180323", "20180324", "20180325", "20180331", "20180402"],
    2019: ["20190321", "20190322", "20190323", "20190324", "20190328", "20190329", "20190330", "20190331", "20190406", "20190408"],
    2021: ["20210319", "20210320", "20210321", "20210322", "20210327", "20210328", "20210329", "20210330", "20210403", "20210405"],
    2022: ["20220317", "20220318", "20220319", "20220320", "20220324", "20220325", "20220326", "20220327", "20220402", "20220404"],
    2023: ["20230316", "20230317", "20230318", "20230319", "20230323", "20230324", "20230325", "20230326", "20230401", "20230403"],
    2024: ["20240321", "20240322", "20240323", "20240324", "20240328", "20240329", "20240330", "20240331", "20240406", "20240408"],
    2025: ["20250318", "20250319", "20250320", "20250321", "20250322", "20250323", "20250327", "20250328", "20250329", "20250330", "20250405", "20250407"],
}

# ESPN team name → Barttorvik team name mapping
ESPN_NAME_MAP = {
    "UConn": "Connecticut",
    "Connecticut Huskies": "Connecticut",
    "NC State": "N.C. State",
    "UNC": "North Carolina",
    "Miami": "Miami FL",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "ETSU": "East Tennessee St.",
    "UCF": "UCF",
    "SMU": "SMU",
    "LSU": "LSU",
    "VCU": "VCU",
    "UAB": "UAB",
    "UNLV": "UNLV",
    "USC": "USC",
    "Pitt": "Pittsburgh",
    "Ole Miss": "Mississippi",
    "St. John's (NY)": "St. John's",
    "Saint Mary's (CA)": "Saint Mary's",
    "Saint Joseph's": "Saint Joseph's",
    "Saint Peter's": "Saint Peter's",
}

ROUND_MAP = {
    64: 64, 32: 32, 16: 16, 8: 8, 4: 4, 2: 2,
    "Round of 64": 64, "Round of 32": 32, "Sweet 16": 16,
    "Elite Eight": 8, "Elite 8": 8, "Final Four": 4,
    "Semifinals": 4, "Championship": 2, "National Championship": 2,
    "First Four": 68,
}


def normalize_team_name(espn_name: str) -> str:
    """Normalize ESPN team name to Barttorvik format."""
    name = espn_name.strip()
    if name in ESPN_NAME_MAP:
        return ESPN_NAME_MAP[name]
    name = re.sub(r"\s+(Wildcats|Bears|Tigers|Bulldogs|Eagles|Hawks|Knights|"
                  r"Panthers|Rams|Warriors|Wolverines|Hoosiers|Jayhawks|"
                  r"Spartans|Buckeyes|Boilermakers|Razorbacks|Commodores|"
                  r"Cavaliers|Cougars|Huskies|Fighting Illini|Cornhuskers|"
                  r"Crimson Tide|Red Raiders|Badgers|Volunteers|Tar Heels|"
                  r"Cardinals|Blue Devils|Gators|Cyclones|Zips|Panthers|"
                  r"Mountaineers|Bruins|Gaels|Hurricanes|Terrapins|Aggies|"
                  r"Horned Frogs|Billikens|Hawkeyes|Broncos|Musketeers|"
                  r"Wolfpack|Longhorns|Mustangs|RedHawks|Dons|Retrievers|"
                  r"Bison|Sharks|Saints|Mountain Hawks|Lancers|Pride|"
                  r"Trojans|Rainbow Warriors|Hawks|Quakers|Raiders|Owls|"
                  r"Vandals|Paladins|Royals|Panthers)$", "", name)
    name = name.strip()
    return name


def fetch_tournament_games_for_year(year: int, session: requests.Session) -> list:
    """Fetch all tournament games for a year using ESPN scoreboard API."""
    dates = TOURNAMENT_DATES.get(year)
    if not dates:
        print(f"  No dates configured for {year}")
        return []

    all_games = []
    seen_game_ids = set()

    for date_str in dates:
        url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
               f"mens-college-basketball/scoreboard?dates={date_str}&groups=100&limit=100")
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception:
            continue

        events = data.get("events", [])
        for event in events:
            game_id = event.get("id")
            if game_id in seen_game_ids:
                continue
            seen_game_ids.add(game_id)

            status = event.get("status", {}).get("type", {}).get("name", "")
            if status != "STATUS_FINAL":
                continue

            competitions = event.get("competitions", [])
            if not competitions:
                continue
            comp = competitions[0]

            # Check if this is NCAA tournament
            notes = comp.get("notes", [])
            is_ncaa = False
            round_name = ""
            for note in notes:
                headline = note.get("headline", "")
                if "NCAA" in headline or "March Madness" in headline or "Championship" in headline:
                    is_ncaa = True
                    round_name = headline
                    break

            # Also check the season type
            season_info = event.get("season", {})
            season_type = season_info.get("type", 0)
            if season_type == 3:
                is_ncaa = True

            if not is_ncaa:
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            for c in competitors:
                team_data = c.get("team", {})
                team_name = team_data.get("location", team_data.get("displayName", ""))
                seed_str = c.get("curatedRank", {}).get("current", 0)
                if seed_str == 0:
                    seed_match = re.search(r"(\d+)", str(c.get("seed", "")))
                    seed_str = int(seed_match.group(1)) if seed_match else 16

                score = int(c.get("score", 0))
                winner = c.get("winner", False)

                all_games.append({
                    "game_id": game_id,
                    "year": year,
                    "team": normalize_team_name(team_name),
                    "seed": int(seed_str) if seed_str else 16,
                    "score": score,
                    "winner": winner,
                    "round_name": round_name,
                    "date": date_str,
                })

        time.sleep(0.5)

    return all_games


def games_to_matchups(all_games: list) -> pd.DataFrame:
    """Convert raw game data to Tournament Matchups format."""
    rows = []
    games_by_id = {}

    for g in all_games:
        gid = g["game_id"]
        if gid not in games_by_id:
            games_by_id[gid] = []
        games_by_id[gid].append(g)

    game_num = 0
    for gid, teams in games_by_id.items():
        if len(teams) != 2:
            continue

        t1, t2 = teams[0], teams[1]
        year = t1["year"]

        # Determine round from date ordering
        date = t1["date"]
        round_name = t1.get("round_name", "")

        # Map round name to numeric round
        current_round = None
        for key, val in ROUND_MAP.items():
            if isinstance(key, str) and key.lower() in round_name.lower():
                current_round = val
                break

        if current_round is None:
            # Infer from date position within tournament
            dates = TOURNAMENT_DATES.get(year, [])
            if date in dates:
                idx = dates.index(date)
                if idx < 2:
                    current_round = 64
                elif idx < 4:
                    current_round = 32
                elif idx < 6:
                    current_round = 16
                elif idx < 8:
                    current_round = 8
                elif idx < 9:
                    current_round = 4
                else:
                    current_round = 2

        if current_round is None:
            current_round = 64

        game_num += 1

        # Determine ROUND (how far each team went)
        winner = t1 if t1["winner"] else t2
        loser = t2 if t1["winner"] else t1

        # Winner's ROUND is at least the next round; loser's ROUND is current_round
        loser_round = current_round
        winner_round = current_round // 2 if current_round > 2 else 1

        for team_data, team_round in [(winner, winner_round), (loser, loser_round)]:
            rows.append({
                "YEAR": year,
                "CURRENT ROUND": current_round,
                "BY ROUND NO": game_num,
                "TEAM": team_data["team"],
                "SEED": team_data["seed"],
                "ROUND": team_round,
                "SCORE": team_data["score"],
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("BUILDING TOURNAMENT MATCHUPS FROM ESPN API")
    print("=" * 70)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    all_games = []
    years = sorted(TOURNAMENT_DATES.keys())

    for i, year in enumerate(years):
        if year == SKIP_YEAR:
            continue
        print(f"[{i+1}/{len(years)}] Fetching {year}...", end=" ", flush=True)
        games = fetch_tournament_games_for_year(year, session)
        print(f"{len(games)//2} games")
        all_games.extend(games)
        time.sleep(1.0)

    print(f"\nTotal raw entries: {len(all_games)}")

    df = games_to_matchups(all_games)
    print(f"Matchup rows: {len(df)}")
    print(f"Years covered: {sorted(df['YEAR'].unique())}")
    print(f"Games per year:")
    for yr in sorted(df['YEAR'].unique()):
        n_games = len(df[df['YEAR'] == yr]) // 2
        print(f"  {yr}: {n_games} games")

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
