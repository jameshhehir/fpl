import requests
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary


# Load data from the API
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
response = requests.get(url)
data = response.json()

# Convert JSON data to DataFrames
elements_df = pd.DataFrame(data['elements'])
element_types_df = pd.DataFrame(data['element_types'])
teams_df = pd.DataFrame(data['teams'])

# Select relevant columns
columns_of_interest = ['id', 'second_name', 'team', 'element_type', 'selected_by_percent', 'now_cost', 'minutes', 'transfers_in', 'value_season', 'total_points']
slim_elements_df = elements_df[columns_of_interest]

# Map element type to position and team ID to team name
slim_elements_df['position'] = slim_elements_df['element_type'].map(element_types_df.set_index('id')['singular_name'])
slim_elements_df['team_name'] = slim_elements_df['team'].map(teams_df.set_index('id')['name'])

# Convert 'value_season' to float
slim_elements_df['value'] = slim_elements_df['value_season'].astype(float)

# Add value per cost column
slim_elements_df['value_per_cost'] = slim_elements_df['total_points'] / slim_elements_df['now_cost']

# Filter out rows where value is greater than 0
slim_elements_df = slim_elements_df[slim_elements_df['value'] > 0]

# Define constraints
budget = 1000  # Total budget (considering cost is in 1/10 of actual value)
team_limit = 3  # Max 3 players from a single team
total_players = 15
formation = {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}

# Create the problem
problem = LpProblem("FPL_Team_Selection", LpMaximize)

# Create a binary variable for each player: 1 if the player is selected, 0 otherwise
player_vars = {player['id']: LpVariable(f"player_{player['id']}", cat=LpBinary) for player in slim_elements_df.to_dict('records')}

# Objective function: maximize total points
problem += lpSum(player['total_points'] * player_vars[player['id']] for player in slim_elements_df.to_dict('records'))

# Constraint: total cost should be less than or equal to the budget
problem += lpSum(player['now_cost'] * player_vars[player['id']] for player in slim_elements_df.to_dict('records')) <= budget

# Constraints: exactly 2 goalkeepers, 5 defenders, 5 midfielders, and 3 forwards
for position, count in formation.items():
    problem += lpSum(player_vars[player['id']] for player in slim_elements_df.to_dict('records') if player['position'] == position) == count

# Constraint: no more than 3 players from the same team
for team in slim_elements_df['team'].unique():
    problem += lpSum(player_vars[player['id']] for player in slim_elements_df.to_dict('records') if player['team'] == team) <= team_limit

# Constraint: total number of players should be 15
problem += lpSum(player_vars[player['id']] for player in slim_elements_df.to_dict('records')) == total_players

# Solve the problem
problem.solve()

# Get the selected players
selected_players = [player for player in slim_elements_df.to_dict('records') if player_vars[player['id']].varValue == 1]

# Convert selected players to DataFrame
selected_team_df = pd.DataFrame(selected_players)

# Display selected team
print("Selected Best 15 Players:")
print(selected_team_df[['second_name', 'team_name', 'position', 'now_cost', 'total_points', 'value_per_cost']])
