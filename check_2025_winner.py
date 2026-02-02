"""Check who actually won the 2025 tournament"""

import pandas as pd

# Load data
df = pd.read_csv('data/raw/KenPom Barttorvik.csv')
teams_2025 = df[df['YEAR'] == 2025]

# ROUND=1 means champion
champion = teams_2025[teams_2025['ROUND'] == 1]

print('='*50)
print('2025 ACTUAL TOURNAMENT RESULT')
print('='*50)

if len(champion) > 0:
    c = champion.iloc[0]
    print(f"CHAMPION: {c['TEAM']}")
    print(f"Seed: {int(c['SEED'])}")
    print(f"KADJ EM: {c['KADJ EM']:.2f}")
    print(f"BARTHAG: {c['BARTHAG']:.3f}")
    
    # Check our prediction rank
    preds = pd.read_csv('results/predictions_2025_blind.csv')
    champ_pred = preds[preds['TEAM'] == c['TEAM']]
    if len(champ_pred) > 0:
        rank = int(champ_pred['RANK'].values[0])
        prob = champ_pred['PROB'].values[0]
        print()
        print('='*50)
        print('MODEL PERFORMANCE')
        print('='*50)
        print(f"We ranked {c['TEAM']} at #{rank}")
        print(f"Probability we assigned: {prob:.3f}")
        if rank == 1:
            print("*** PERFECT PREDICTION! ***")
        elif rank <= 5:
            print("*** IN TOP 5 - GOOD! ***")
        elif rank <= 10:
            print("*** IN TOP 10 - DECENT ***")
        else:
            print(f"*** MISSED - Champion was ranked #{rank} ***")
else:
    print("No champion data found (ROUND=1)")
    print("Checking other indicators...")
    
    # Show teams that went furthest
    print("\nTeams by tournament round (lowest ROUND = furthest):")
    furthest = teams_2025.nsmallest(5, 'ROUND')[['TEAM', 'SEED', 'ROUND']]
    print(furthest.to_string(index=False))
