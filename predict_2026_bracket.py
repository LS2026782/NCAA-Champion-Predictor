"""
2026 NCAA Tournament Champion Predictions.

Uses the official bracket from Selection Sunday (March 15, 2026) and
end-of-season team statistics from Barttorvik/KenPom.

Runs two prediction modes:
  1. Champion Predictor — ranks all 68 teams by "champion-likeness"
  2. Bracket Simulation — simulates every game round-by-round
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from config.settings import MODEL_FEATURES
from src.features.builder import FeatureBuilder
from src.models.champion_model import ChampionPredictor, compute_era_weights

# ============================================================
# 2026 NCAA TOURNAMENT BRACKET — Official (Selection Sunday, March 15 2026)
# Team names use Barttorvik format
# ============================================================

BRACKET_2026 = {
    # 1-seeds
    'Duke': 1, 'Arizona': 1, 'Michigan': 1, 'Florida': 1,
    # 2-seeds
    'Houston': 2, 'Connecticut': 2, 'Iowa St.': 2, 'Purdue': 2,
    # 3-seeds
    'Michigan St.': 3, 'Illinois': 3, 'Gonzaga': 3, 'Virginia': 3,
    # 4-seeds
    'Nebraska': 4, 'Alabama': 4, 'Kansas': 4, 'Arkansas': 4,
    # 5-seeds
    'Vanderbilt': 5, "St. John's": 5, 'Texas Tech': 5, 'Wisconsin': 5,
    # 6-seeds
    'Tennessee': 6, 'North Carolina': 6, 'Louisville': 6, 'BYU': 6,
    # 7-seeds
    'Kentucky': 7, "Saint Mary's": 7, 'Miami FL': 7, 'UCLA': 7,
    # 8-seeds
    'Clemson': 8, 'Villanova': 8, 'Ohio St.': 8, 'Georgia': 8,
    # 9-seeds
    'Utah St.': 9, 'TCU': 9, 'Saint Louis': 9, 'Iowa': 9,
    # 10-seeds
    'Santa Clara': 10, 'UCF': 10, 'Missouri': 10, 'Texas A&M': 10,
    # 11-seeds (includes First Four)
    'N.C. State': 11, 'Texas': 11, 'SMU': 11, 'Miami OH': 11,
    'VCU': 11, 'South Florida': 11,
    # 12-seeds
    'McNeese St.': 12, 'Akron': 12, 'High Point': 12, 'Northern Iowa': 12,
    # 13-seeds
    'Cal Baptist': 13, 'Hofstra': 13, 'Troy': 13, 'Hawaii': 13,
    # 14-seeds
    'North Dakota St.': 14, 'Penn': 14, 'Wright St.': 14, 'Kennesaw St.': 14,
    # 15-seeds
    'Idaho': 15, 'Furman': 15, 'Queens': 15, 'Tennessee St.': 15,
    # 16-seeds (includes First Four)
    'Siena': 16, 'LIU': 16, 'Howard': 16, 'UMBC': 16,
    'Prairie View A&M': 16, 'Lehigh': 16,
}

# First Four matchups: (team_a, team_b, seed, region, slot_index)
# slot_index = position in the region's team list where the winner is inserted
FIRST_FOUR = [
    ('UMBC', 'Howard', 16, 'Midwest', 1),           # Tue 3/17 — 16-seed → Midwest
    ('Texas', 'N.C. State', 11, 'West', 9),          # Tue 3/17 — 11-seed → West
    ('Prairie View A&M', 'Lehigh', 16, 'South', 1),  # Wed 3/18 — 16-seed → South
    ('Miami OH', 'SMU', 11, 'Midwest', 9),            # Wed 3/18 — 11-seed → Midwest
]

# Regional bracket assignments (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
# First Four slots use None as placeholder until simulation fills them.
REGIONS = {
    'East': [
        ('Duke', 1), ('Siena', 16),
        ('Ohio St.', 8), ('TCU', 9),
        ("St. John's", 5), ('Northern Iowa', 12),
        ('Kansas', 4), ('Cal Baptist', 13),
        ('Louisville', 6), ('South Florida', 11),
        ('Michigan St.', 3), ('North Dakota St.', 14),
        ('UCLA', 7), ('UCF', 10),
        ('Connecticut', 2), ('Furman', 15),
    ],
    'South': [
        ('Florida', 1), (None, 16),              # First Four: Prairie View A&M vs Lehigh
        ('Clemson', 8), ('Iowa', 9),
        ('Vanderbilt', 5), ('McNeese St.', 12),
        ('Nebraska', 4), ('Troy', 13),
        ('North Carolina', 6), ('VCU', 11),
        ('Illinois', 3), ('Penn', 14),
        ("Saint Mary's", 7), ('Texas A&M', 10),
        ('Houston', 2), ('Idaho', 15),
    ],
    'West': [
        ('Arizona', 1), ('LIU', 16),
        ('Villanova', 8), ('Utah St.', 9),
        ('Wisconsin', 5), ('High Point', 12),
        ('Arkansas', 4), ('Hawaii', 13),
        ('BYU', 6), (None, 11),                  # First Four: Texas vs NC State
        ('Gonzaga', 3), ('Kennesaw St.', 14),
        ('Miami FL', 7), ('Missouri', 10),
        ('Purdue', 2), ('Queens', 15),
    ],
    'Midwest': [
        ('Michigan', 1), (None, 16),              # First Four: UMBC vs Howard
        ('Georgia', 8), ('Saint Louis', 9),
        ('Texas Tech', 5), ('Akron', 12),
        ('Alabama', 4), ('Hofstra', 13),
        ('Tennessee', 6), (None, 11),              # First Four: Miami OH vs SMU
        ('Virginia', 3), ('Wright St.', 14),
        ('Kentucky', 7), ('Santa Clara', 10),
        ('Iowa St.', 2), ('Tennessee St.', 15),
    ],
}


def get_team_data_name(bracket_name: str, available_teams: set) -> str | None:
    """Map bracket name to available team name in the dataset."""
    if bracket_name in available_teams:
        return bracket_name
    bracket_lower = bracket_name.lower()
    for team in available_teams:
        if team.lower() == bracket_lower:
            return team
        if len(bracket_lower) >= 6 and len(team.lower()) >= 6:
            if bracket_lower.startswith(team.lower()) or team.lower().startswith(bracket_lower):
                return team
    return None


# ============================================================
# LOAD DATA
# ============================================================

kp_2026 = pd.read_csv('data/raw/KenPom Barttorvik 2026.csv')
available_teams = set(kp_2026['TEAM'].values)

print('=' * 70)
print('2026 MARCH MADNESS PREDICTIONS')
print('Official bracket (Selection Sunday, March 15 2026)')
print('=' * 70)

# Build bracket dataset from the 68 tournament teams
bracket_teams = []
for bracket_name, seed in BRACKET_2026.items():
    data_name = get_team_data_name(bracket_name, available_teams)
    if data_name:
        team_row = kp_2026[kp_2026['TEAM'] == data_name].iloc[0].to_dict()
        team_row['BRACKET_NAME'] = bracket_name
        team_row['PROJ_SEED'] = seed
        team_row['SEED'] = seed
        bracket_teams.append(team_row)

bracket_df = pd.DataFrame(bracket_teams)
print(f'\nMatched {len(bracket_df)} of {len(BRACKET_2026)} bracket teams to data')

# ============================================================
# CHAMPION PREDICTOR
# ============================================================
print('\n' + '=' * 70)
print('CHAMPION PREDICTOR: Who looks most "champion-like"?')
print('=' * 70)

kp_historical = pd.read_csv('data/raw/KenPom Barttorvik Extended.csv')
kp_train = kp_historical[kp_historical['YEAR'] < 2026].copy()
kp_train['IS_CHAMPION'] = (kp_train['ROUND'] == 1).astype(int)
kp_train = kp_train[kp_train['SEED'].notna()].copy()

bracket_df['YEAR'] = 2026
bracket_df['ROUND'] = 64
bracket_df['IS_CHAMPION'] = 0

combined = pd.concat([kp_train, bracket_df], ignore_index=True)

builder = FeatureBuilder()
_, X_combined = builder.build_features(combined, fit_scaler=True)

n_train = len(kp_train)
X_train_scaled = X_combined[:n_train]
X_bracket_scaled = X_combined[n_train:]
y_train = kp_train['IS_CHAMPION'].values

season_groups = kp_train['YEAR'].values
era_weights = compute_era_weights(season_groups)
model = ChampionPredictor(model_type='logreg', calibrate=True)
model.fit(X_train_scaled, y_train, feature_names=builder.get_feature_names(),
          season_groups=season_groups, era_weights=era_weights)

bracket_df['CHAMP_PROB'] = model.predict_proba(X_bracket_scaled)

# Normalize so probabilities sum to 1.0 across the field
prob_sum = bracket_df['CHAMP_PROB'].sum()
if prob_sum > 0:
    bracket_df['CHAMP_PROB'] = bracket_df['CHAMP_PROB'] / prob_sum

champ_ranked = bracket_df.sort_values('CHAMP_PROB', ascending=False)

print('\nTop 20 "Champion-Like" Teams:')
print('-' * 70)
for i, (_, row) in enumerate(champ_ranked.head(20).iterrows(), 1):
    print(f"{i:2d}. ({int(row['PROJ_SEED'])}) {row['BRACKET_NAME']:20s} | "
          f"Prob: {row['CHAMP_PROB']*100:.1f}% | EM: {row['KADJ EM']:+.1f}")

# ============================================================
# GAME-BY-GAME BRACKET SIMULATION
# ============================================================
print('\n' + '=' * 70)
print('BRACKET SIMULATION: Predicting each game')
print('=' * 70)

try:
    from src.models.game_predictor import GamePredictor, GAME_FEATURES
    from src.data.game_loader import GameLoader

    loader = GameLoader()
    games_df, team_stats = loader.load_all()
    predictor = GamePredictor(model_type='logreg')
    predictor.fit(games_df, team_stats)
except (FileNotFoundError, ImportError, Exception) as e:
    print(f"\nBracket simulation skipped: {e}")
    print("(Champion probability rankings above are the primary prediction output.)")
    import sys
    sys.exit(0)


def simulate_game(
    team_a: str, seed_a: int,
    team_b: str, seed_b: int,
    predictor: GamePredictor, team_stats: pd.DataFrame
) -> tuple[str, int, float]:
    """Simulate a single game. Returns (winner_name, winner_seed, win_prob)."""
    stats_available = set(team_stats['TEAM'].values)
    name_a = get_team_data_name(team_a, stats_available)
    name_b = get_team_data_name(team_b, stats_available)

    if not name_a or not name_b:
        if seed_a <= seed_b:
            return (team_a, seed_a, 0.65)
        return (team_b, seed_b, 0.65)

    stats_a = team_stats[team_stats['TEAM'] == name_a].iloc[-1]
    stats_b = team_stats[team_stats['TEAM'] == name_b].iloc[-1]

    diff = []
    for col in GAME_FEATURES:
        val_a = stats_a.get(col, 0) if col in stats_a.index else 0
        val_b = stats_b.get(col, 0) if col in stats_b.index else 0
        if pd.isna(val_a):
            val_a = 0
        if pd.isna(val_b):
            val_b = 0
        diff.append(val_a - val_b)

    X = np.array([diff])
    X_scaled = predictor.scaler.transform(X)
    prob_a = predictor.model.predict_proba(X_scaled)[0][1]

    if prob_a > 0.5:
        return (team_a, seed_a, prob_a)
    return (team_b, seed_b, 1 - prob_a)


def simulate_region(
    region_teams: list, region_name: str,
    predictor: GamePredictor, team_stats: pd.DataFrame
) -> tuple[str, int]:
    """Simulate a region through to the Final Four."""
    print(f'\n{region_name} Region:')
    current_round = list(region_teams)
    round_names = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8']

    for round_num, round_name in enumerate(round_names):
        next_round = []
        for i in range(0, len(current_round), 2):
            team_a, seed_a = current_round[i]
            team_b, seed_b = current_round[i + 1]
            winner, winner_seed, prob = simulate_game(
                team_a, seed_a, team_b, seed_b, predictor, team_stats
            )
            next_round.append((winner, winner_seed))
            if round_num >= 2:
                print(f"  {round_name}: ({seed_a}){team_a} vs ({seed_b}){team_b} "
                      f"-> ({winner_seed}){winner} ({prob:.0%})")
        current_round = next_round

    return current_round[0]


# ── Simulate First Four and fill regional brackets ─────────────────
print('\nFIRST FOUR:')
resolved_regions: dict[str, list[tuple[str, int]]] = {
    r: list(teams) for r, teams in REGIONS.items()
}

for team_a, team_b, seed, region, slot_idx in FIRST_FOUR:
    winner, _, prob = simulate_game(team_a, seed, team_b, seed, predictor, team_stats)
    print(f"  ({seed}){team_a} vs ({seed}){team_b} -> {winner} ({prob:.0%})")
    resolved_regions[region][slot_idx] = (winner, seed)

print('\nSimulating bracket (showing Sweet 16 and beyond)...')

final_four = []
for region_name, teams in resolved_regions.items():
    winner = simulate_region(teams, region_name, predictor, team_stats)
    final_four.append((winner[0], winner[1], region_name))
    print(f"  -> {region_name} CHAMPION: ({winner[1]}) {winner[0]}")

# Final Four
print('\n' + '=' * 70)
print('FINAL FOUR')
print('=' * 70)

# Semifinal 1: East vs South
sf1 = simulate_game(
    final_four[0][0], final_four[0][1],
    final_four[1][0], final_four[1][1],
    predictor, team_stats,
)
print(f"Semifinal 1: ({final_four[0][1]}){final_four[0][0]} [{final_four[0][2]}] "
      f"vs ({final_four[1][1]}){final_four[1][0]} [{final_four[1][2]}]")
print(f"  -> Winner: ({sf1[1]}) {sf1[0]} ({sf1[2]:.0%})")

# Semifinal 2: West vs Midwest
sf2 = simulate_game(
    final_four[2][0], final_four[2][1],
    final_four[3][0], final_four[3][1],
    predictor, team_stats,
)
print(f"Semifinal 2: ({final_four[2][1]}){final_four[2][0]} [{final_four[2][2]}] "
      f"vs ({final_four[3][1]}){final_four[3][0]} [{final_four[3][2]}]")
print(f"  -> Winner: ({sf2[1]}) {sf2[0]} ({sf2[2]:.0%})")

# Championship
print('\n' + '=' * 70)
print('CHAMPIONSHIP GAME')
print('=' * 70)
champ = simulate_game(sf1[0], sf1[1], sf2[0], sf2[1], predictor, team_stats)
print(f"({sf1[1]}) {sf1[0]} vs ({sf2[1]}) {sf2[0]}")
print(f"\n*** PREDICTED 2026 CHAMPION: ({champ[1]}) {champ[0]} ({champ[2]:.0%}) ***")
