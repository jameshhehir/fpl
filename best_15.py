import requests
import pandas as pd
import numpy as np



# Load data from the API
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
response = requests.get(url)
data = response.json()

# Convert JSON data to DataFrames
elements_df = pd.DataFrame(data['elements'])
element_types_df = pd.DataFrame(data['element_types'])
teams_df = pd.DataFrame(data['teams'])

# Select relevant columns
columns_of_interest = ['second_name', 'team', 'element_type', 'selected_by_percent', 'now_cost', 'minutes', 'transfers_in', 'value_season', 'total_points']
slim_elements_df = elements_df[columns_of_interest]

# Map element type to position and team ID to team name
slim_elements_df['position'] = slim_elements_df['element_type'].map(element_types_df.set_index('id')['singular_name'])
slim_elements_df['team'] = slim_elements_df['team'].map(teams_df.set_index('id')['name'])

# Convert 'value_season' to float
slim_elements_df['value'] = slim_elements_df['value_season'].astype(float)

# Add value per cost column
slim_elements_df['value_per_cost'] = slim_elements_df['total_points'] / slim_elements_df['now_cost']

# Filter out rows where value is greater than 0
slim_elements_df = slim_elements_df[slim_elements_df['value'] > 0]

# Define budget and constraints
budget = 1000  # Total budget (considering cost is in 1/10 of actual value)
team_limit = 3  # Max 3 players from a single team

# Desired formation for starting XI
formation = {'Goalkeeper': 1, 'Defender': 4, 'Midfielder': 4, 'Forward': 2}

# Initialize team
selected_team = []

# Function to select players for a position
def select_players(position, num_players, available_budget, current_team_count, selected_player_ids):
    candidates = slim_elements_df[(slim_elements_df['position'] == position) & (~slim_elements_df.index.isin(selected_player_ids))]
    candidates = candidates.sort_values(by=['value_per_cost', 'total_points'], ascending=False)
    selected = []
    team_count = current_team_count.copy()

    for _, player in candidates.iterrows():
        if len(selected) >= num_players:
            break
        if available_budget - player['now_cost'] < 0:
            continue
        if team_count.get(player['team'], 0) >= team_limit:
            continue

        selected.append(player)
        available_budget -= player['now_cost']
        team_count[player['team']] = team_count.get(player['team'], 0) + 1
        selected_player_ids.add(player.name)

    return selected, available_budget, team_count

# Select players for starting XI
remaining_budget = budget
team_count = {}
selected_player_ids = set()

for position, num_players in formation.items():
    selected_players, remaining_budget, team_count = select_players(position, num_players, remaining_budget, team_count, selected_player_ids)
    selected_team.extend(selected_players)

# Define bench positions
bench_positions = {'Goalkeeper': 1, 'Defender': 1, 'Midfielder': 1, 'Forward': 1}

# Select players for the bench
for position, num_players in bench_positions.items():
    selected_players, remaining_budget, team_count = select_players(position, num_players, remaining_budget, team_count, selected_player_ids)
    selected_team.extend(selected_players)

# Convert selected team to DataFrame
selected_team_df = pd.DataFrame(selected_team)

# Display selected team
print("Selected Best 15 Players:")
print(selected_team_df[['second_name', 'team', 'position', 'now_cost', 'total_points', 'value_per_cost']])
