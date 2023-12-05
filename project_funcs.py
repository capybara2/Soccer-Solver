'''
Functions for John Knight final project
MATH-6340 Prof. Mike Lindstrom
'''

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Add another option to run analysis on lots of seasons; can do this by
# creating a loop that creates instances of Season for each then combine results?
# (Leave this until the end; get everything else finished first)

# Need to annotate the make_chart method when finished.


    
def create_season():
    
    '''
    Prompts the user to choose a league and season, then creates a Season instance.
    
    Returns:
    season (Season): A Season class object.
    '''
    while True: # Loop allows for error handling
        
        print("1. Premier League (England)\n2. La Liga (Spain)\n3. Bundesliga (Germany)\n"
              "4. Serie A (Italy)\n5. Ligue 1 (France)\n")
        league = input("Choose a league from the above options: ")
        if league in ["1", "2", "3", "4", "5"]:
            league_id, league_name = choice_to_fbref(league) # Converts to fbref format
            year = input("Available seasons are as follows:\n"
                           "2017-2018\n"
                           "2018-2019\n"
                           "2019-2020\n"
                           "2020-2021\n"
                           "2021-2022\n"
                           "2022-2023\n"
                           "\n"
                           "Enter a season, or leave blank for current season:\n")
            if year in ["2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022",
                          "2022-2023", ""]:
                season = Season(league_id, league_name, year)
                return season
        print("\nInvalid choice entered!\n") # Then repeats the while loop

def choice_to_fbref(league):
    
    '''
    Converts the numbered choice of league to the ID and name used by the fbref website.
    
    Parameters:
    choice (str): the option entered by the user.
    
    Returns:
    fbref_id_name (tuple) the ID and league name used by fbref.com.
    '''
    leagues = {'1':('9', 'Premier-League'), '2':('12', 'La-Liga'), '3':('20', 'Bundesliga'),
                     '4':('11', 'Serie-A'), '5':('13', 'Ligue-1')}
    
    fbref_id_name = leagues[league]
    return fbref_id_name

def scrape_scores(league_id, league_name, season=None):
    
    '''
    Uses the libraries 'requests' and 'beautifulsoup' to scrape match results
    from fbref.com for the given league and season.
    
    Parameters:
    league_id (str) the league id number in fbref.com format.
    league_name (str) the league name in hyphenated fbref.com format.
    season (str) a season formatted as YYYY-YYYY, or None if using current season.
    
    Returns:
    df (pandas DataFrame object) a data frame containing match data.
    '''

    # Different url format for current season vs. previous seasons:
    if season:
        url = "https://fbref.com/en/comps/" + league_id + "/" + season +  "/schedule/" + season + "-" + league_name + "-Scores-and-Fixtures"
    else: # Season == None means current season
        url = "https://fbref.com/en/comps/" + league_id + "/schedule/" + league_name + "-Scores-and-Fixtures"
    
    print("Fetching data from", url)
    
    r = requests.get(url) # gets the html
    soup = BeautifulSoup(r.content, 'html.parser') # Converts html to BS object
    
    table = soup.find('table') # Scores & Fixtures is always the first table

    # Convert the table into a pandas DataFrame
    df = pd.read_html(str(table), header=0)[0]
    
    # Drop rows where Score == 'Score' or xG = 'xG' (header column is repeated on fbref.com)
    df = df[df['Score'] != 'Score']
    
    # Rename 'xG' and 'xG.1' columns to xGH and xGA
    df = df.rename(columns={'xG': 'xGH', 'xG.1': 'xGA'})
    df['xGH'] = df['xGH'].astype(float)
    df['xGA'] = df['xGA'].astype(float)
    
    # Drop rows where 'Score' is NaN (these are games that have not been played)
    df = df.dropna(subset=['Score'])
    df = df.reset_index(drop=True) # drop=True drops the old index
    
    # Use regex to remove penalthy shootout scores in parentheses
    df['Score'] = df['Score'].str.replace(r"\s*\(\d+\)", "", regex=True)

    # Split 'Score' into home and away scores by the separating hyphen
    df[['H_Score', 'A_Score']] = df['Score'].str.split('–', expand=True) # expand=True creates separate columns
    df['H_Score'] = df['H_Score'].astype(int)
    df['A_Score'] = df['A_Score'].astype(int)
    
    # Set Date column to type Datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    return df
        
def solve(df, metric, end_date=None):
    
    '''
    Uses linear optimization to solve for the coefficients for each team rating and home advantage.
    
    Parameters:
    df (pandas DataFrame object): A data frame containing the columns Home, Away,
    H_Score, A_Score, xGH, xGA (plus possible other columns which won't affect the calculations).
    metric (str): '1' for Goals or '2' for Expected Goals.
    end_date (Date): A Datetime object representing the end date to be analyzed (non-inclusive)
    
    Returns:
    team_ratings_dict (dict): A dictionary mapping teams to their ratings.
    sorted_team_ratings (list): A list of tuples of the teams and their ratings in descending order.
    home_advantage (float): The optimized value for home advantage.
    latest_date (Datetime object): The latest date of the included matches.
    '''
    
    # Check if df is empty to avoid error.
    if len(df) == 0:
        return {}, None, None, None
    
    # Remove rows where Date >= end_date
    if end_date:
        df = df[df['Date'] < end_date]
    
    # If using xG, remove rows where xGH or xGA is NaN
    if metric == '2':
        df = df.dropna(subset=['xGH'])
        df = df.dropna(subset=['xGA'])
        df = df.reset_index(drop=True) # drop=True drops the old index
    
    # Create a mapping of teams to indices
    teams = pd.unique(df[['Home', 'Away']].values.ravel()) # ravel turns 2D array into 1D
    team_to_index = {team: i for i, team in enumerate(teams)} # creates dict of (i, team) items
    
    m = len(df) # Number of matches
    n = len(teams) # Number of teams
    
    # Set up the coefficients matrix and results vector
    
    # Each row in matrix corresponds to a match; columns correspond to teams and home advantage
    coeffs_matrix = np.zeros((m, n + 1))  # Plus 1 for home advantage column
      
    # If metric == '1', use goal difference. Else, expected goal difference.
    if metric == '1':
        goal_diff = df['H_Score'] - df['A_Score']
    else:
        goal_diff = df['xGH'] - df['xGA']
    
    # Set the coefficients matrix
    for idx, row in df.iterrows():
        coeffs_matrix[idx, team_to_index[row['Home']]] = 1  # Home team coefficient
        coeffs_matrix[idx, team_to_index[row['Away']]] = -1  # Away team coefficient
        coeffs_matrix[idx, -1] = 1  # Home advantage coefficient
    
    # Objective function: Minimize the sum of absolute differences
    
    # Need to add 2*n auxiliary variables for absolute values
    c = np.concatenate([np.zeros(n + 1), np.ones(2 * m)])  # Minimize these auxiliary variables
    
    # Extend coeffs_matrix to handle absolute differences
    extended_coeffs_matrix = np.hstack([coeffs_matrix, -np.eye(m), np.eye(m)])  
    
    # Set up constraints
    A_eq = extended_coeffs_matrix
    b_eq = goal_diff
    
    # Add a constraint ensuring the mean team rating is zero
    # This row has 1s for each team rating, 0s for the home advantage and auxiliary variables
    mean_zero_constraint = np.concatenate([np.ones(n), np.zeros(1 + 2 * m)])  # 1 for home advantage, 2*n for auxiliary variables
    A_eq = np.vstack([A_eq, mean_zero_constraint])
    # Add the corresponding value (0) to b_eq
    b_eq = np.append(b_eq, 0)
    
    x_bounds = (-5, 5) # Team ratings should not be outside this range
    bounds = [x_bounds] * (n + 1) + [(0, None)] * (2 * m)  # Auxiliary variables are non-negative
    
    # Solve the linear program
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    team_ratings = result.x[:n]
    home_advantage = result.x[n]

    # Create a dictionary mapping each team to its rating
    team_ratings_dict = {team: rating for team, rating in zip(teams, team_ratings)}
    
    # Sort the teams by their ratings in descending order
    sorted_team_ratings = sorted(team_ratings_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Find the latest date in the dataset
    latest_date = df['Date'].max()
    
    return team_ratings_dict, sorted_team_ratings, home_advantage, latest_date



class Season:
    
    '''
    An object representing a distinct season of a given competition.
    '''
    
    def __init__(self, league_id, league_name, year):
        
        '''
        Class constructor function. Creates the class data frame by calling scrape_scores()
        
        Parameters:
        league_id (str): The fbref-formatted id number of the league.
        league_name (str): The fbref-formatted name of the league (with hyphens).
        year (str): The season in format YYYY-YYYY.
        '''
        
        self.__league_id = league_id
        self.__league_name = league_name
        self.__league_unhyphen = league_name.replace('-', ' ')
        self.__year = None if year == "" else year # None means current season used
        self.__df = scrape_scores(league_id, league_name, year)
        
    def main_menu(self):
        
        '''
        Prompts the user to choose current ratings or perform analysis, then runs
        the methods of the chosen option.
        '''
        
        while True: # While loop allows error handling.
            choice = input("1. Get current ratings\n"
                       "2. Perform analysis\n"
                       "\n"
                       "Choose an option: ")
            if (choice == '1'):
                self.get_current_ratings()
                return
            elif (choice == '2'):
                self.get_all_ratings()
                self.__calc_abs_diffs()
                self.__make_chart()
                filename = "Ratings " + self.__league_name + " " + (self.__year if self.__year else "Current Season") + ".csv"
                self.__df.to_csv(filename, index=False) # Output the data frame to file.
                return
            else:
                print("Invalid choice. Please choose again.\n\n")
            
    
    def get_current_ratings(self):
        
        '''
        Asks the user to choose goals or expected goals, then solves for ratings
        using that metric using the class instance data frame. Prints the results
        to both the console and a .txt file.
        '''
        
        while True: # loop allows error handling
            metric = input("Enter 1 to use goals, or 2 to use expected goals: ")
            if metric not in ["1", "2"]:
                print("\nInvalid choice. Please choose again.\n")
            else:
                team_ratings_dict, sorted_team_ratings, home_advantage, latest_date = solve(self.__df, metric)
                formatted_date = latest_date.strftime("%B %d, %Y") # For printing
            
                # Open a file to write the output
                with open("ratings_output.txt", "w") as file:
                    print("\nHere are the current ratings up to and including games on", formatted_date, ":\n")
                    file.write(f"\nHere are the current ratings up to and including games on {formatted_date} :\n\n")
            
                    for team, rating in sorted_team_ratings:
                        output_line = f"{team} {rating:.2f}\n"
                        print(output_line, end='')
                        file.write(output_line)
            
                    home_advantage_output = f"\nHome advantage: {home_advantage:.2f}\n"
                    print(home_advantage_output)
                    file.write(home_advantage_output)
                return
        

            
    
    def get_all_ratings(self):
        
        '''
        Iterates through each row in the class instance data frame and calculates
        the ratings for each team as it would have been on that date, i.e. only
        including matches played before that date. It does this for both G and xG.
        '''
        
        
        # Create new columns for the game-by-game ratings
        self.__df['H_Rating_G'] = None
        self.__df['A_Rating_G'] = None
        self.__df['Home_Adv_G'] = None
        self.__df['H_Rating_xG'] = None
        self.__df['A_Rating_xG'] = None
        self.__df['Home_Adv_xG'] = None
        
        # Iterate through each game and create retrospective ratings
        for i, row in self.__df.iterrows():
            print("Calculating match", i+1, "of", len(self.__df))
            # Goals
            team_ratings_dict, sorted_team_ratings, home_advantage, latest_date = solve(self.__df, metric='1', end_date=row['Date'])
            if row['Home'] in team_ratings_dict: # Won't be in dict if they haven't played yet
                self.__df.at[i, 'H_Rating_G'] = team_ratings_dict[row['Home']]
            if row['Away'] in team_ratings_dict:
                self.__df.at[i, 'A_Rating_G'] = team_ratings_dict[row['Away']]
            self.__df.at[i, 'Home_Adv_G'] = home_advantage
            # Expected Goals
            team_ratings_dict, sorted_team_ratings, home_advantage, latest_date = solve(self.__df, metric='2', end_date=row['Date'])
            if row['Home'] in team_ratings_dict:
                self.__df.at[i, 'H_Rating_xG'] = team_ratings_dict[row['Home']]
            if row['Away'] in team_ratings_dict:
                self.__df.at[i, 'A_Rating_xG'] = team_ratings_dict[row['Away']]
            self.__df.at[i, 'Home_Adv_xG'] = home_advantage
    
    def __calc_abs_diffs(self):
        
        '''
        Calculates the absolute difference between the forecasted score and the actual score,
        using both G and xG as the predictor, then prints the overall mean of both.
        '''
        
        self.__df['GD'] = self.__df['H_Score'] - self.__df['A_Score']
        self.__df['Forecast_GD'] = self.__df['H_Rating_G'] - self.__df['A_Rating_G'] + self.__df['Home_Adv_G']
        self.__df['Forecast_xGD'] = self.__df['H_Rating_xG'] - self.__df['A_Rating_xG'] + self.__df['Home_Adv_xG']
        self.__df['Abs_Diff_G'] = abs(self.__df['Forecast_GD'] - self.__df['GD'])
        self.__df['Abs_Diff_xG'] = abs(self.__df['Forecast_xGD'] - self.__df['GD'])
        # Note that GD is used as the target variable even when xG is used as the predictor,
        # since goals are the ultimate currency of football that we care about predicting.
        
        # Print the overall mean abs diff for G and xG.
        mean_G = round(np.mean(self.__df['Abs_Diff_G']), 3)
        mean_xG = round(np.mean(self.__df['Abs_Diff_xG']), 3)
        print("Mean absolute difference using G:", mean_G)
        print("Mean absolute difference using xG:", mean_xG)
    
    def __make_chart(self, rolling_n=5): 
        
        '''
        Makes a line chart showing the rolling 5-game mean absolute difference
        between predicted score and actual score, with one line showing Goals as the
        predictor and the other showing xG. Also prints the overall mean abs diff for both.
        '''
        
        # Sort __df by Date.
        
    
        # Create a DataFrame that stacks home and away games in one column then sort by date
        home_games = self.__df[['Date', 'Home']].rename(columns={'Home': 'Team'})
        away_games = self.__df[['Date', 'Away']].rename(columns={'Away': 'Team'})
        all_games = pd.concat([home_games, away_games])
        all_games = all_games.sort_values(by='Date')
    
        # Get cumulative count of games played by each team on each date
        all_games['N'] = all_games.groupby('Team').cumcount() + 1
    
        # Merge this count back into the original DataFrame for home and away games
        self.__df = self.__df.merge(all_games, left_on=['Date', 'Home'], right_on=['Date', 'Team'], how='left').rename(columns={'N': 'Home_N'})
        self.__df = self.__df.merge(all_games, left_on=['Date', 'Away'], right_on=['Date', 'Team'], how='left').rename(columns={'N': 'Away_N'})
    
        # Now combine home and away data including N into a single df
        home_data = self.__df[['Date', 'Home_N', 'Abs_Diff_G', 'Abs_Diff_xG']].rename(columns={'Home_N': 'N'})
        away_data = self.__df[['Date', 'Away_N', 'Abs_Diff_G', 'Abs_Diff_xG']].rename(columns={'Away_N': 'N'})
        combined_data = pd.concat([home_data, away_data])
    
        # Calculate rolling averages
        combined_data['Rolling_Avg_G'] = combined_data.groupby('Date')['Abs_Diff_G'].transform(lambda x: x.rolling(window=rolling_n, min_periods=1).mean())
        combined_data['Rolling_Avg_xG'] = combined_data.groupby('Date')['Abs_Diff_xG'].transform(lambda x: x.rolling(window=rolling_n, min_periods=1).mean())
    
        # Calculate the mean rolling averages for all teams at each value of N
        mean_rolling_avgs = combined_data.groupby('N')[['Rolling_Avg_G', 'Rolling_Avg_xG']].mean()
    
        # Filter out rows where N is less than rolling_n
        mean_rolling_avgs = mean_rolling_avgs[mean_rolling_avgs.index >= rolling_n]
        
        # Make the plot
        plt.figure(figsize=(10, 6))
        plt.plot(mean_rolling_avgs.index, mean_rolling_avgs['Rolling_Avg_G'], label='G')
        plt.plot(mean_rolling_avgs.index, mean_rolling_avgs['Rolling_Avg_xG'], label='xG')
        plt.xlabel('Games Played')
        plt.ylabel('Mean Absolute Difference')
        plt.title(self.__league_unhyphen + ' ' + (self.__year if self.__year else "Current Season") + ' Mean Rolling ' + str(rolling_n) + ' Game Absolute Difference for G and xG')
        plt.ylim(0)
        plt.legend()
        
        # Save the plot
        filename = "Rolling avg " + self.__league_unhyphen + ' ' + (self.__year if self.__year else "Current Season") + ".png"
        plt.savefig(filename)  # Saves the plot as a PNG file in the current working directory
    
        plt.show()
        


