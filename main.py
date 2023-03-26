import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import requests
from Model import train_model, predict_odds
import pandas as pd

# Define the URL to fetch data from
url = 'https://api.hockeychallengehelper.com/api/picks'

# Set a flag to determine whether to use the API or not
apiavaliability = False

# If the API is available, fetch the data from the URL and convert the response to JSON format
if apiavaliability == True:
    response = requests.get(url)
    json_data = response.json()

    # Extract the full names of all the players from the JSON data
    full_names = []
    for player_list in json_data['playerLists']:
        for player in player_list['players']:
            full_names.append(player['fullName'])

    # Divide the list of full names into three parts of roughly equal size
    quotient, remainder = divmod(len(full_names), 3)
    base1 = full_names[:quotient + (1 if remainder > 0 else 0)]
    base2 = full_names[quotient + (1 if remainder > 1 else 0):2 * quotient + (2 if remainder > 0 else 0)]
    base3 = full_names[2 * quotient + (2 if remainder > 1 else 0):]

    # Print the three subsets of player names
    print(base1)
    print(base2)
    print(base3)

# Define three lists of NHL players
base1 = ['Connor McDavid', 'Sidney Crosby', 'Quinn Hughes']
base2 = ['Connor McDavid', 'Leon Draisaitl', 'Mitchell Marner']
base3 = ['Jack Eichel', 'Jack Hughes', 'Matthew Tkachuk']

# Set the base URL for the NHL API
base_url = "https://statsapi.web.nhl.com/api/v1"

# Initialize three empty lists
part1 = []
part2 = []
part3 = []

# Send a GET request to the teams endpoint of the NHL API and load the response into a JSON object
response = requests.get(f"{base_url}/teams")
teams = json.loads(response.text)

# Initialize an empty list to store player information
player_info = []


# Iterate through each team in the response from the NHL API
for team in teams["teams"]:
    # Get the team ID and name from the team object
    team_id = team["id"]
    team_name = team["name"]

    # Set the endpoint for retrieving the roster for this team
    roster_endpoint = f"/teams/{team_id}/roster"

    # Send a GET request to the roster endpoint of this team and load the response into a JSON object
    response = requests.get(base_url + roster_endpoint)
    roster = json.loads(response.text)

    # Iterate through each player on this team's roster
    for player in roster["roster"]:
        # Get the player ID and name from the player object
        player_id = player["person"]["id"]
        player_name = player["person"]["fullName"]

        # Check if the player's name is in base1
        if player_name in base1:
            # Append the player's ID to part1
            part1.append(player_id)

        # Check if the player's name is in base2
        elif player_name in base2:
            # Append the player's ID to part2
            part2.append(player_id)

        # Check if the player's name is in base3
        elif player_name in base3:
            # Append the player's ID to part3
            part3.append(player_id)


print("Part 1:", part1)
print("Part 2:", part2)
print("Part 3:", part3)

lists = [part1, part2, part3]

# Initialize an empty list to store the results
all_player_info = []

# Loop through each list
for player_list in lists:

    # Initialize an empty list to store player information
    player_info = []

    # Loop through each player in the list
    for i in player_list:


        api_url = f'https://statsapi.web.nhl.com/api/v1/people/{i}/stats?season=20222023&stats=gameLog'

        # Send request to API and extract JSON data
        response = requests.get(api_url)
        json_data = response.json()

        # Extract relevant data and create a DataFrame
        rows = []
        for split in json_data['stats'][0]['splits']:
            try:
                shooting_pct = split['stat']['shotPct']
            except KeyError:
                shooting_pct = 0.0
            opp_id = split['opponent']['id']
            opp_url = f'https://statsapi.web.nhl.com/api/v1/teams/{opp_id}?expand=team.stats&season=20222023'
            opp_response = requests.get(opp_url)
            opp_data = opp_response.json()
            opp_ga_per_gp = opp_data['teams'][0]['teamStats'][0]['splits'][0]['stat']['goalsAgainstPerGame']
            row = {
                'date': split['date'],
                'goals': split['stat']['goals'],
                'shots': split['stat']['shots'],
                'shooting_pct': shooting_pct,
                'opp_id': opp_id,
                'opp_name': split['opponent']['name'],
                'opp_ga_per_gp': opp_ga_per_gp,
                'scored': 1 if split['stat']['goals'] > 0 else 0,
            }
            rows.append(row)

        mcDavid_data = pd.DataFrame(rows)

        # Save DataFrame to CSV
        mcDavid_data.to_csv('mcdavid_2019_data.csv', index=False)

        gpg = f'https://statsapi.web.nhl.com/api/v1/people/{i}/stats?stats=statsSingleSeason&season=20222023'
        question = requests.get(gpg)
        json_data = question.json()
        games = json_data["stats"][0]["splits"][0]["stat"]["games"]
        shtpct = json_data["stats"][0]["splits"][0]["stat"]["shotPct"]

        # Set up the data
        goals = json_data["stats"][0]["splits"][0]["stat"]["goals"]
        gp = json_data["stats"][0]["splits"][0]["stat"]["games"]
        shooting_pct = float(json_data["stats"][0]["splits"][0]["stat"]["shotPct"])
        shots = float(json_data["stats"][0]["splits"][0]["stat"]["shots"]/gp)
        gpg = int(goals)/gp
        opp_gaa = 3  # hypothetical opponent goals against average per game
        mcDavid_data = pd.read_csv('mcdavid_2019_data.csv')
        model, scaler = train_model('mcdavid_2019_data.csv')

        # Preprocess the data
        odds = predict_odds(model, scaler, gpg, shots, shooting_pct, opp_gaa, 'mcdavid_2019_data.csv')

        player_fullnames = f'https://statsapi.web.nhl.com/api/v1/people/{i}'
        response = requests.get(player_fullnames)
        json_data = response.json()
        full_name = json_data['people'][0]['fullName']

        print(f'The predicted odds of {str(full_name)} scoring in a hypothetical game are {odds:.2%}')
        player_info.append({'name': full_name, 'odds': odds})

    player_info = sorted(player_info, key=lambda x: x['odds'], reverse=True)

    # Print the chart
    print('Player\t\tPredicted Odds')
    print('--------------------------------')
    for info in player_info:
        print(f'{info["name"]}\t{info["odds"]:.2%}')

