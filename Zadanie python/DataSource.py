import requests
import logging
import re
import pandas as pd


class Clubs():
    """Class responsible for fetching and processing information about football clubs
    participating in competitions from a specified API endpoint."""

    # Base URL of the API from where the data is fetched.
    url = 'http://localhost:8000/'

    def __init__(self, comp_id, season_id_start, season_id_end):
        """
        Initializes the Clubs instance with competition ID and the range of seasons.

        Parameters:
        - comp_id (str): The competition ID to fetch clubs for.
        - season_id_start (int): The starting season ID for fetching data.
        - season_id_end (int): The ending season ID for fetching data.
        """
        self.comp_id = comp_id
        self.season_id_start = season_id_start
        self.season_id_end = season_id_end
        self.logger = logging.getLogger(__name__)

        # Setup basic logging configuration.
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    def clubs_competitions(self):
        """
        Fetches competition data for clubs within the specified season range.

        Returns:
        - A collection of club data aggregated across the specified seasons.
        """
        try:
            for i in range(int(self.season_id_start), int(self.season_id_end) + 1):
                season = str(i)
                # Append each season's URL to the list.
                url_competitions = self.url + 'competitions/' + self.comp_id + '/clubs/' + '?season_id=' + season
                response = requests.get(url_competitions)
                data = response.json()
                competitions_id= data['id']
                data= data['clubs']
                for club in data:
                    club.update({'Competitions': competitions_id})
                # Aggregate club data, initializing with the first season's data.
                if i == int(self.season_id_start):
                    club_dict = data
                else:
                    club_dict += data
            self.logger.info("Club data sourced")
            return club_dict
        except requests.RequestException as e:
            self.logger.error(f'Club data sourcing failed. Error: {e}')
            return None

    def clubs_ids(self):
        """
        Extracts and returns a list of unique club IDs from the competition data.

        Returns:
        - A list of unique club IDs.
        """
        list_comp = self.clubs_competitions()
        clubs_ids = [club['id'] for club in list_comp]
        # Ensure the list is unique.
        clubs_ids = list(set(clubs_ids))
        self.logger.info('List of clubs ids created')
        return clubs_ids

    def club_dataframe(self):
        """
        Creates a DataFrame with unique club names and IDs from the competition data.

        Returns:
        - A pandas DataFrame containing columns for club names and IDs.
        """
        list_comp = self.clubs_competitions()
        club_names = pd.DataFrame(list_comp)
        # Remove duplicate entries.
        club_names = club_names.drop_duplicates(ignore_index=True)
        self.logger.info('Reference data for club names and ids created')
        return club_names

    def __iter__(self):
        """
        Makes the class iterable over the club IDs, allowing for easy iteration through club IDs.

        Returns:
        - An iterator over the club IDs.
        """
        return iter(self.clubs_ids())


class Players(Clubs):
    """
    A class that extends Clubs to handle player-specific data,
    including player details, transfers, and profiles.
    """

    def __init__(self, comp_id, season_id_start, season_id_end):
        """
        Initialize the Players class by calling the initializer of Clubs.
        """
        super().__init__(comp_id, season_id_start, season_id_end)
    
    def create_players_dict(self):
        """
        Fetches and aggregates players' data from all clubs for the specified competition seasons.
        Returns a list of player data dictionaries.
        """
        try:
            clubs = self.clubs_ids()  # Fetch club IDs
            urls_clubs = []
            # Generate URLs for fetching player data for each club and season
            for i in range(int(self.season_id_start), int(self.season_id_end) + 1):
                season = str(i)
                urls_clubs += [f'{self.url}clubs/{club_id}/players?season_id={season}' for club_id in clubs]
            
            players_dict = []
            # Fetch player data from each URL
            for url in urls_clubs:
                response = requests.get(url)
                data = response.json()
                players_dict.append(data)
            self.logger.info('Players data loaded to json')
            return players_dict
        except requests.RequestException as e:
            self.logger.error(f'Players data loading to json failed. Error: {e}')
            return None
        
    def create_players_df(self):
        """
        Converts the players' data into a pandas DataFrame.
        Returns a DataFrame with player data.
        """
        try:
            players = self.create_players_dict()
            data_frame = []
            # Aggregate all players into a single list
            for element in players:
                data_frame += element['players']
            # Convert to DataFrame
            final_df = pd.DataFrame(data_frame)
            # Extract the first nationality if multiple are present
            final_df['nationality'] = final_df['nationality'].apply(lambda x: x[0])
            # Remove duplicate entries
            final_df = final_df.drop_duplicates(ignore_index=True)
            self.logger.info('DataFrame created')
            return final_df
        except requests.RequestException as e:
            self.logger.error(f'Players DataFrame creation failed. Error: {e}')
            return None

    def transfers_df(self, create=True, path=None):
        """
        Fetches and constructs a DataFrame of player transfers.
        Can either create a new DataFrame of players or use an existing one from a CSV file.
        Returns a DataFrame with transfer data.
        """
        try:
            if create:
                players = self.create_players_df()
            else:
                players = pd.read_csv(path)
            players = list(players['id'])
            transfers_list = []
            # Fetch transfer data for each player
            for player_id in players:
                url = f'{self.url}players/{player_id}/transfers'
                response = requests.get(url)
                data = response.json()
                transfers_list.append(data)
            list_dfs = []
            # Construct a DataFrame for each player's transfers
            for transfer in transfers_list:
                df = pd.DataFrame(transfer['transfers'])
                df['id'] = transfer['id']
                # Simplify club ID references
                df['from'] = df['from'].apply(lambda x: x['clubID'])
                df['to'] = df['to'].apply(lambda x: x['clubID'])
                list_dfs.append(df)
            # Concatenate all DataFrames
            df = pd.concat(list_dfs)
            self.logger.info('Transfers DataFrame created')
            return df
        except requests.RequestException as e:
            self.logger.error(f'Transfers DataFrame creation failed. Error: {e}')
            return None
        
    def profile_df(self, create=True, path=None):
        """
        Fetches and constructs a DataFrame of players' profiles.
        Depending on the `create` flag, it either generates a new DataFrame of players or uses an existing one from a CSV file.
        
        Parameters:
        - create (bool): Flag to determine whether to create a new DataFrame or use an existing one.
        - path (str, optional): Path to the CSV file if `create` is False.
        
        Returns:
        - A DataFrame containing players' profile information.
        """
        try:    
            if create:
                players = self.create_players_df()
            else:
                players = pd.read_csv(path)
            players = list(players['id'])
            profile_urls = [f'{self.url}players/{i}/profile' for i in players]
            profile_list = []
            # Fetch profile data for each player
            for url in profile_urls:
                response = requests.get(url)
                data = response.json()
                profile_list.append(data)
            list_dfs = []
            # Process and consolidate profile data into DataFrames
            for index, element in enumerate(profile_list):
                key_columns = ['id', 'age', 'citizenship', 'position']
                element = {k: element[k] for k in key_columns}
                if isinstance(element['citizenship'], list):
                    element['citizenship'] = element['citizenship'][0]
                if isinstance(element['position'], dict):
                    element['position'] = element['position']['main']
                df = pd.DataFrame([element])
                list_dfs.append(df)
            df = pd.concat(list_dfs)
            self.logger.info('Profiles DataFrame created')
            return df
        except requests.RequestException as e:
            self.logger.error(f'Profile DataFrame creation failed. Error: {e}')
            return None

    def stats_df(self, create=True, path=None):
        """
        Fetches and constructs a DataFrame of players' statistics.
        Depending on the `create` flag, it either generates a new DataFrame of players or uses an existing one from a CSV file.
        
        Parameters:
        - create (bool): Flag to determine whether to create a new DataFrame or use an existing one.
        - path (str, optional): Path to the CSV file if `create` is False.
        
        Returns:
        - A DataFrame containing players' statistics.
        """
        try:
            if create:
                players = self.create_players_df()
            else:
                players = pd.read_csv(path)
            players = list(players['id'])
            stats_urls = [f'{self.url}players/{i}/stats' for i in players]
            stats_list = []
            # Fetch statistics data for each player
            for url in stats_urls:
                response = requests.get(url)
                data = response.json()
                stats_list.append(data)
            list_dfs = []
            # Convert season format and consolidate statistics data into DataFrames
            pattern = r"(\d{2})/\d{2}"
            def replace_year(match):
                return "20" + match.group(1)
            def transform_season_to_year(season):
                return re.sub(pattern, replace_year, season)
            for index, element in enumerate(stats_list):
                df = pd.DataFrame(element['stats'])
                df['id'] = element['id']
                if 'seasonID' in df.columns:
                    df['seasonID'] = df['seasonID'].apply(transform_season_to_year)
                list_dfs.append(df)
            df = pd.concat(list_dfs)
            self.logger.info('Stats DataFrame created')
            return df
        except requests.RequestException as e:
            self.logger.error(f'Stats DataFrame creation failed. Error: {e}')
            return None



        








 




    

    
  