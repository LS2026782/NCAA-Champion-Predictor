"""
Fun prediction for Andy Katz's 2026 projected bracket.
Using 2025 team stats as proxy (most recent available).
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from config.settings import MODEL_FEATURES
from src.models.game_predictor import GamePredictor, GAME_FEATURES
from src.features.builder import FeatureBuilder
from src.models.champion_model import ChampionPredictor, compute_era_weights

# Andy Katz's 2026 bracket predictions (from NCAA.com Feb 3, 2026)
BRACKET_2026 = {
    # 1-seeds
    'Arizona': 1, 'Michigan': 1, 'Duke': 1, 'UConn': 1,
    # 2-seeds
    'Nebraska': 2, 'Houston': 2, 'Iowa State': 2, 'Illinois': 2,
    # 3-seeds
    'Michigan State': 3, 'Gonzaga': 3, 'Purdue': 3, 'Kansas': 3,
    # 4-seeds
    'Texas Tech': 4, 'Florida': 4, 'Vanderbilt': 4, 'BYU': 4,
    # 5-seeds
    'Virginia': 5, 'North Carolina': 5, 'Louisville': 5, 'Tennessee': 5,
    # 6-seeds
    "St. John's": 6, 'Alabama': 6, 'Arkansas': 6, 'Kentucky': 6,
    # 7-seeds
    'Iowa': 7, 'Saint Louis': 7, 'Clemson': 7, 'Auburn': 7,
    # 8-seeds
    'UCF': 8, 'Texas A&M': 8, 'Villanova': 8, 'SMU': 8,
    # 9-seeds
    'Wisconsin': 9, 'NC State': 9, 'Utah State': 9, 'Indiana': 9,
    # 10-seeds
    "Saint Mary's": 10, 'Georgia': 10, 'USC': 10, 'Miami': 10,
    # 11-seeds
    'UCLA': 11, 'New Mexico': 11, 'Ohio State': 11, 'San Diego State': 11,
    'Texas': 11, 'Miami (Ohio)': 11,
    # 12-seeds
    'Belmont': 12, 'Tulsa': 12, 'Liberty': 12, 'Yale': 12,
    # 13-seeds
    'Stephen F. Austin': 13, 'UNC Wilmington': 13, 'High Point': 13, 'Utah Valley': 13,
    # 14-seeds
    'North Dakota State': 14, 'UC Irvine': 14, 'Troy': 14, 'Austin Peay': 14,
    # 15-seeds
    'Portland State': 15, 'Wright State': 15, 'UT Martin': 15, 'East Tennessee State': 15,
    # 16-seeds
    'Navy': 16, 'Merrimack': 16, 'LIU': 16, 'Bethune-Cookman': 16, 
    'Vermont': 16, 'Maryland Eastern Shore': 16,
}

# Name mappings (bracket names to data names)
NAME_MAP = {
    'UConn': 'Connecticut',
    "St. John's": "St. John's",
    "Saint Mary's": "Saint Mary's",
    'NC State': 'North Carolina St.',
    'Miami': 'Miami FL',
    'Miami (Ohio)': 'Miami OH',
    'SMU': 'SMU',
    'Iowa State': 'Iowa St.',
    'Michigan State': 'Michigan St.',
    'East Tennessee State': 'ETSU',
    'Stephen F. Austin': 'SF Austin',
    'North Dakota State': 'North Dakota St.',
    'Utah Valley': 'Utah Valley St.',
    'Maryland Eastern Shore': 'Md.-Eastern Shore',
}

def get_team_name(bracket_name, available_teams):
    """Map bracket name to available team name — exact and close matches only."""
    if bracket_name in NAME_MAP:
        mapped = NAME_MAP[bracket_name]
        if mapped in available_teams:
            return mapped
    if bracket_name in available_teams:
        return bracket_name
    # Partial match: only accept if one is a strict prefix/suffix of the other (not a substring)
    bracket_lower = bracket_name.lower()
    for team in available_teams:
        team_lower = team.lower()
        # Require the match to be at word boundaries — both must be ≥ 6 chars to avoid false positives
        if len(team_lower) >= 6 and len(bracket_lower) >= 6:
            if team_lower == bracket_lower:
                return team
            if bracket_lower.startswith(team_lower) or team_lower.startswith(bracket_lower):
                return team
    return None

# Load 2026 data (actual current season stats!)
kp = pd.read_csv('data/raw/KenPom Barttorvik 2026.csv')
kp_2025 = kp.copy()  # Use 2026 data directly

# Add seed strength from projected seeds (will be overwritten per team below)
kp_2025['SEED_NUM'] = kp_2025['SEED'].fillna(16).astype(float)
kp_2025['SEED_STRENGTH'] = 17 - kp_2025['SEED_NUM']

# EM_X_SOS will be recomputed after loading training data to ensure consistent normalization
kp_2025['EM_X_SOS'] = 0  # placeholder

available_teams = set(kp_2025['TEAM'].values)

print('='*70)
print('2026 MARCH MADNESS PREDICTIONS (Fun Speculation!)')
print('Based on Andy Katz bracket projection + 2025 team stats as proxy')
print('='*70)

# Build bracket dataset
bracket_teams = []
for bracket_name, seed in BRACKET_2026.items():
    data_name = get_team_name(bracket_name, available_teams)
    if data_name:
        team_row = kp_2025[kp_2025['TEAM'] == data_name].iloc[0].to_dict()
        team_row['BRACKET_NAME'] = bracket_name
        team_row['PROJ_SEED'] = seed
        team_row['SEED_NUM'] = seed  # Use projected seed
        team_row['SEED_STRENGTH'] = 17 - seed
        bracket_teams.append(team_row)

bracket_df = pd.DataFrame(bracket_teams)
print(f'\nMatched {len(bracket_df)} of {len(BRACKET_2026)} teams to 2025 data')

# ============================================================
# CHAMPION PREDICTOR
# ============================================================
print('\n' + '='*70)
print('CHAMPION PREDICTOR: Who looks most "champion-like"?')
print('='*70)

# Train on historical data using Extended CSV (complete features, 2002-2025)
kp_historical = pd.read_csv('data/raw/KenPom Barttorvik Extended.csv')
kp_train = kp_historical[kp_historical['YEAR'] < 2026].copy()
kp_train['IS_CHAMPION'] = (kp_train['ROUND'] == 1).astype(int)
kp_train = kp_train[kp_train['SEED'].notna()].copy()

# Add YEAR to bracket_df so FeatureBuilder can compute RELATIVE_3PR per-season
bracket_df['YEAR'] = 2026
bracket_df['ROUND'] = 64  # placeholder (unknown)
bracket_df['IS_CHAMPION'] = 0

# Combine train + bracket so FeatureBuilder sees 2026 teams when computing season averages
combined = pd.concat([kp_train, bracket_df], ignore_index=True)

# Use the proper FeatureBuilder (handles EM_X_SOS normalization, RELATIVE_3PR, etc.)
builder = FeatureBuilder()
_, X_combined = builder.build_features(combined, fit_scaler=True)

n_train = len(kp_train)
X_train_scaled = X_combined[:n_train]
X_bracket_scaled = X_combined[n_train:]
y_train = kp_train['IS_CHAMPION'].values

# Train calibrated model
season_groups = kp_train['YEAR'].values
era_weights = compute_era_weights(season_groups)
model = ChampionPredictor(model_type='logreg', calibrate=True)
model.fit(X_train_scaled, y_train, feature_names=builder.get_feature_names(),
          season_groups=season_groups, era_weights=era_weights)

bracket_df['CHAMP_PROB'] = model.predict_proba(X_bracket_scaled)

# Rank and display
champ_ranked = bracket_df.sort_values('CHAMP_PROB', ascending=False)

print('\nTop 15 "Champion-Like" Teams:')
print('-'*70)
for i, (_, row) in enumerate(champ_ranked.head(15).iterrows(), 1):
    print(f"{i:2d}. ({int(row['PROJ_SEED'])}) {row['BRACKET_NAME']:20s} | "
          f"Prob: {row['CHAMP_PROB']*100:.2f}% | EM: {row['KADJ EM']:+.1f}")

# ============================================================
# GAME-BY-GAME BRACKET SIMULATION
# ============================================================
print('\n' + '='*70)
print('BRACKET SIMULATION: Predicting each game')
print('='*70)

# Train game predictor on all historical games
from src.data.game_loader import GameLoader
try:
    loader = GameLoader()
    games_df, team_stats = loader.load_all()
except FileNotFoundError as e:
    print(f"\nBracket simulation skipped: {e}")
    print("(Champion probability rankings above are the primary prediction output.)")
    import sys; sys.exit(0)

predictor = GamePredictor(model_type='logreg')
predictor.fit(games_df, team_stats)

# Set up bracket regions (from article)
regions = {
    'West': [('Arizona', 1), ('Vermont', 16), ('Villanova', 8), ('Indiana', 9),
             ('Virginia', 5), ('Belmont', 12), ('Vanderbilt', 4), ('UNC Wilmington', 13),
             ('Arkansas', 6), ('UCLA', 11), ('Gonzaga', 3), ('North Dakota State', 14),
             ('Saint Louis', 7), ('Miami', 10), ('Nebraska', 2), ('Portland State', 15)],
    'South': [('UConn', 1), ('Merrimack', 16), ('Texas A&M', 8), ('NC State', 9),
              ('Tennessee', 5), ('Yale', 12), ('BYU', 4), ('Stephen F. Austin', 13),
              ("St. John's", 6), ('San Diego State', 11), ('Michigan State', 3), ('UC Irvine', 14),
              ('Iowa', 7), ('Georgia', 10), ('Houston', 2), ('East Tennessee State', 15)],
    'Midwest': [('Michigan', 1), ('LIU', 16), ('SMU', 8), ('Utah State', 9),
                ('North Carolina', 5), ('Tulsa', 12), ('Texas Tech', 4), ('Utah Valley', 13),
                ('Kentucky', 6), ('Miami (Ohio)', 11), ('Purdue', 3), ('Austin Peay', 14),
                ('Auburn', 7), ('USC', 10), ('Iowa State', 2), ('Wright State', 15)],
    'East': [('Duke', 1), ('Navy', 16), ('UCF', 8), ('Wisconsin', 9),
             ('Louisville', 5), ('Liberty', 12), ('Florida', 4), ('High Point', 13),
             ('Alabama', 6), ('Ohio State', 11), ('Kansas', 3), ('Troy', 14),
             ('Clemson', 7), ("Saint Mary's", 10), ('Illinois', 2), ('UT Martin', 15)],
}

def simulate_game(team_a, seed_a, team_b, seed_b, bracket_df, predictor, team_stats):
    """Simulate a single game and return winner."""
    # Get team data
    name_a = get_team_name(team_a, set(team_stats['TEAM'].values))
    name_b = get_team_name(team_b, set(team_stats['TEAM'].values))
    
    if not name_a or not name_b:
        # Default to better seed
        if seed_a <= seed_b:
            return (team_a, seed_a, 0.65)
        else:
            return (team_b, seed_b, 0.65)
    
    # Get most recent stats
    stats_a = team_stats[team_stats['TEAM'] == name_a].iloc[-1]
    stats_b = team_stats[team_stats['TEAM'] == name_b].iloc[-1]
    
    # Build features
    diff = []
    for col in GAME_FEATURES:
        val_a = stats_a.get(col, 0) if col in stats_a.index else 0
        val_b = stats_b.get(col, 0) if col in stats_b.index else 0
        if pd.isna(val_a): val_a = 0
        if pd.isna(val_b): val_b = 0
        diff.append(val_a - val_b)
    
    X = np.array([diff])
    X_scaled = predictor.scaler.transform(X)
    prob_a = predictor.model.predict_proba(X_scaled)[0][1]
    
    if prob_a > 0.5:
        return (team_a, seed_a, prob_a)
    else:
        return (team_b, seed_b, 1 - prob_a)

def simulate_region(region_teams, region_name, bracket_df, predictor, team_stats):
    """Simulate a region through to Final Four."""
    print(f'\n{region_name} Region:')
    
    current_round = region_teams.copy()
    round_names = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8']
    
    for round_num, round_name in enumerate(round_names):
        next_round = []
        for i in range(0, len(current_round), 2):
            team_a, seed_a = current_round[i][0], current_round[i][1]
            team_b, seed_b = current_round[i+1][0], current_round[i+1][1]
            
            result = simulate_game(team_a, seed_a, team_b, seed_b, bracket_df, predictor, team_stats)
            winner, winner_seed, prob = result
            next_round.append((winner, winner_seed))
            
            if round_num >= 2:  # Show Sweet 16 and Elite 8
                print(f"  {round_name}: ({seed_a}){team_a} vs ({seed_b}){team_b} -> ({winner_seed}){winner} ({prob:.0%})")
        
        current_round = next_round
    
    return current_round[0]  # Regional winner

print('\nSimulating bracket (showing Sweet 16 and beyond)...')

final_four = []
for region_name, teams in regions.items():
    winner = simulate_region(teams, region_name, bracket_df, predictor, team_stats)
    final_four.append((winner[0], winner[1], region_name))
    print(f"  -> {region_name} CHAMPION: ({winner[1]}) {winner[0]}")

# Final Four
print('\n' + '='*70)
print('FINAL FOUR')
print('='*70)

# Semifinal 1: West vs South
sf1 = simulate_game(final_four[0][0], final_four[0][1], final_four[1][0], final_four[1][1], 
                    bracket_df, predictor, team_stats)
print(f"Semifinal 1: ({final_four[0][1]}){final_four[0][0]} vs ({final_four[1][1]}){final_four[1][0]}")
print(f"  -> Winner: ({sf1[1]}) {sf1[0]} ({sf1[2]:.0%})")

# Semifinal 2: Midwest vs East
sf2 = simulate_game(final_four[2][0], final_four[2][1], final_four[3][0], final_four[3][1],
                    bracket_df, predictor, team_stats)
print(f"Semifinal 2: ({final_four[2][1]}){final_four[2][0]} vs ({final_four[3][1]}){final_four[3][0]}")
print(f"  -> Winner: ({sf2[1]}) {sf2[0]} ({sf2[2]:.0%})")

# Championship
print('\n' + '='*70)
print('CHAMPIONSHIP GAME')
print('='*70)
champ = simulate_game(sf1[0], sf1[1], sf2[0], sf2[1], bracket_df, predictor, team_stats)
print(f"({sf1[1]}) {sf1[0]} vs ({sf2[1]}) {sf2[0]}")
print(f"\n*** PREDICTED 2026 CHAMPION: ({champ[1]}) {champ[0]} ({champ[2]:.0%}) ***")
