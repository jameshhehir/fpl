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

# Display top 10 players by value
top_10_by_value = slim_elements_df.sort_values('value', ascending=False).head(10)
print(top_10_by_value)

# Pivot table for average value by position
position_pivot = slim_elements_df.pivot_table(index='position', values='value', aggfunc=np.mean).reset_index()
print(position_pivot.sort_values('value', ascending=False))

# Filter out rows where value is greater than 0
slim_elements_df = slim_elements_df[slim_elements_df['value'] > 0]

# Pivot table for average value by team
team_pivot = slim_elements_df.pivot_table(index='team', values='value', aggfunc=np.mean).reset_index()
print(team_pivot.sort_values('value', ascending=False))

# Filter DataFrame by position
fwd_df = slim_elements_df[slim_elements_df['position'] == 'Forward']
mid_df = slim_elements_df[slim_elements_df['position'] == 'Midfielder']
def_df = slim_elements_df[slim_elements_df['position'] == 'Defender']
goal_df = slim_elements_df[slim_elements_df['position'] == 'Goalkeeper']

# Display top 10 players by value for each position
print("Top Forwards:")
print(fwd_df.sort_values('value', ascending=False).head(10))

print("Top Midfielders:")
print(mid_df.sort_values('value', ascending=False).head(10))

print("Top Defenders:")
print(def_df.sort_values('value', ascending=False).head(10))

print("Top Goalkeepers:")
print(goal_df.sort_values('value', ascending=False).head(10))
