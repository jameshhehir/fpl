import requests
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
import warnings

warnings.filterwarnings('ignore')

# Load data from the FPL API
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
response = requests.get(url)
data = response.json()

# Convert JSON data to DataFrames
elements_df = pd.DataFrame(data['elements'])
element_types_df = pd.DataFrame(data['element_types'])
teams_df = pd.DataFrame(data['teams'])

# Select relevant columns
columns_of_interest = ['id', 'second_name', 'team', 'element_type', 'selected_by_percent', 'now_cost', 'minutes', 'transfers_in', 'value_season', 'total_points', 'form']
slim_elements_df = elements_df[columns_of_interest]

# Map element type to position and team ID to team name
slim_elements_df['position'] = slim_elements_df['element_type'].map(element_types_df.set_index('id')['singular_name'])
slim_elements_df['team_name'] = slim_elements_df['team'].map(teams_df.set_index('id')['name'])

# Convert 'value_season' and 'form' to float
slim_elements_df['value'] = slim_elements_df['value_season'].astype(float)
slim_elements_df['form'] = slim_elements_df['form'].astype(float)
slim_elements_df['selected_by_percent'] = slim_elements_df['selected_by_percent'].astype(float)

# Add value per cost and expected points (simple heuristic combining total points and form)
slim_elements_df['value_per_cost'] = slim_elements_df['total_points'] / slim_elements_df['now_cost']
slim_elements_df['expected_points'] = 0.7 * slim_elements_df['total_points'] + 0.3 * slim_elements_df['form'] * 10

# Filter out players with zero expected points
slim_elements_df = slim_elements_df[slim_elements_df['expected_points'] > 0]

# Define constraints
budget = 1000  # Total budget (considering cost is in 1/10 of actual value)
team_limit = 3  # Max 3 players from a single team
total_players = 15
formation = {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}

# Create the problem
problem = LpProblem("FPL_Team_Selection", LpMaximize)

# Create a binary variable for each player: 1 if the player is selected, 0 otherwise
player_vars = {player['id']: LpVariable(f"player_{player['id']}", cat=LpBinary) for player in slim_elements_df.to_dict('records')}

# Objective function: maximize expected points and popularity (selected_by_percent)
problem += lpSum((player['expected_points'] + 0.1 * player['selected_by_percent']) * player_vars[player['id']] for player in slim_elements_df.to_dict('records'))

# Constraint: total cost should be less than or equal to the budget
problem += lpSum(player['now_cost'] * player_vars[player['id']] for player in slim_elements_df.to_dict('records')) <= budget

# Constraints: exactly 2 goalkeepers, 5 defenders, 5 midfielders, and 3 forwards
for position, count in formation.items():
    problem += lpSum(player_vars[player['id']] for player in slim_elements_df.to_dict('records') if player['position'] == position) == count

# Constraint: no more than 3 players from the same team
for team in slim_elements_df['team_name'].unique():
    problem += lpSum(player_vars[player['id']] for player in slim_elements_df.to_dict('records') if player['team_name'] == team) <= team_limit

# Constraint: total number of players should be 15
problem += lpSum(player_vars[player['id']] for player in slim_elements_df.to_dict('records')) == total_players

# Additional constraints for more complexity

# Ensure balanced budget distribution
max_budget_per_position = {'Goalkeeper': 100, 'Defender': 250, 'Midfielder': 350, 'Forward': 300}
for position, max_budget in max_budget_per_position.items():
    problem += lpSum(player['now_cost'] * player_vars[player['id']] for player in slim_elements_df.to_dict('records') if player['position'] == position) <= max_budget

# Ensure positional value
min_points_per_position = {'Goalkeeper': 100, 'Defender': 200, 'Midfielder': 300, 'Forward': 250}
for position, min_points in min_points_per_position.items():
    problem += lpSum(player['total_points'] * player_vars[player['id']] for player in slim_elements_df.to_dict('records') if player['position'] == position) >= min_points

# Create binary variables for captain and vice-captain
captain_var = {player_id: LpVariable(f"captain_{player_id}", cat=LpBinary) for player_id in slim_elements_df['id']}
vice_captain_var = {player_id: LpVariable(f"vice_captain_{player_id}", cat=LpBinary) for player_id in slim_elements_df['id']}

# Captain and vice-captain must be selected players
for player_id in slim_elements_df['id']:
    problem += captain_var[player_id] <= player_vars[player_id]
    problem += vice_captain_var[player_id] <= player_vars[player_id]

# Only one captain and one vice-captain
problem += lpSum(captain_var[player_id] for player_id in slim_elements_df['id']) == 1
problem += lpSum(vice_captain_var[player_id] for player_id in slim_elements_df['id']) == 1

# Captain and vice-captain cannot be the same player
for player_id in slim_elements_df['id']:
    problem += captain_var[player_id] + vice_captain_var[player_id] <= 1

# Create auxiliary variables for captain and vice-captain contributions
captain_contrib = LpVariable.dicts("captain_contrib", slim_elements_df['id'], lowBound=0)
vice_captain_contrib = LpVariable.dicts("vice_captain_contrib", slim_elements_df['id'], lowBound=0)

# Link auxiliary variables to the corresponding player's expected points
for player_id in slim_elements_df['id']:
    player_expected_points = slim_elements_df.loc[slim_elements_df['id'] == player_id, 'expected_points'].values[0]
    problem += captain_contrib[player_id] == player_expected_points * captain_var[player_id]
    problem += vice_captain_contrib[player_id] == player_expected_points * vice_captain_var[player_id]

# Adjust the objective function to include captain and vice-captain contributions
problem += lpSum((player['expected_points'] * player_vars[player['id']] +
                  0.5 * captain_contrib[player['id']] +
                  0.25 * vice_captain_contrib[player['id']] +
                  0.1 * player['selected_by_percent']) 
                 for player in slim_elements_df.to_dict('records'))

# Solve the problem
problem.solve()

# Get the selected players
selected_players = [player for player in slim_elements_df.to_dict('records') if player_vars[player['id']].varValue == 1]

# Convert selected players to DataFrame
selected_team_df = pd.DataFrame(selected_players)

# Display selected team
print("Selected Best 15 Players:")
print(selected_team_df[['second_name', 'team_name', 'position', 'now_cost', 'total_points', 'expected_points', 'value_per_cost']])

# Get captain and vice-captain
captain = next(player for player in selected_players if captain_var[player['id']].varValue == 1)
vice_captain = next(player for player in selected_players if vice_captain_var[player['id']].varValue == 1)

print("\nCaptain:")
print(captain)

print("\nVice-Captain:")
print(vice_captain)
