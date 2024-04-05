import requests
import logging
import aiohttp
import asyncio
import re
import pandas as pd


class Clubs():

    url= 'http://localhost:8000/'
    def __init__(self, comp_id, season_id_start, season_id_end):
        self.comp_id= comp_id
        self.season_id_start= season_id_start
        self.season_id_end= season_id_end
        self.logger = logging.getLogger(__name__) 

        logging.basicConfig(level=logging.INFO,  # Ustawia minimalny poziom logowania
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Format wyświetlania logu
                    datefmt='%Y-%m-%d %H:%M:%S')  # Format daty i czasu


    def clubs_competitions (self):
        try:
            url_competitions=[]
            for i in range (int(self.season_id_start), int(self.season_id_end)+1):
                season =str(i)
                url_competitions+= self.url + 'competitions/'+ self.comp_id + '/clubs/' + '?season_id=' + season
                response = requests.get(url_competitions)
                data= response.json()
                data= data['clubs']
                if i == int(self.season_id_start):
                    club_dict= data
                else:
                    club_dict += data  
            self.logger.info("Club data sourced")
            return competitions
        except requests.RequestException as e:
            self.logger.error(f'Club data sourcing failed. Error: {e}')
            return None
        
    
    def clubs_ids (self):
        list_comp= self.clubs_competitions()
        clubs_ids= [i for i in list_comp]
        clubs_ids = [club['id'] for club in clubs_ids]
        clubs_ids= list(set(clubs_ids))
        self.logger.info  ('List of clubs ids created')
        return clubs_ids
    
    def club_names (self):
        list_comp= self.clubs_competitions()
        club_names= pd.DataFrame(list_comp)
        club_names= club_names.drop_duplicates(ignore_index=True)
        self.logger.info  ('Reference data for club names and ids created')
        return club_names
    
    def __iter__ (self):
        return (iter(self.clubs_ids()))


class Players(Clubs):

    def __init__(self, comp_id, season_id_start, season_id_end):
        super().__init__(comp_id, season_id_start, season_id_end)   
    
    def create_players_dict(self):
        try:
            club_ids = self.clubs_ids()
            for i in range (int(self.season_id_start), int(self.season_id_end)+1):
                season =str(i)
                urls_clubs_list = [f'{self.url}clubs/{club_id}/players?season_id={season}' for club_id in club_ids]
                if i == int(self.season_id_start):
                    urls_clubs= urls_clubs_list
                else:
                    urls_clubs += urls_clubs_list 
            players_dict= []
            for i in urls_clubs:
                response= requests.get(i)
                data= response.json()
                players_dict.append(data)
            self.logger.info ('Players data loaded to json')
            return players_dict
        except requests.RequestException as e:
            self.logger.error (f'Players data loading to json failed. Error: {e}')
            return None
        
    def create_players_df(self):
        try:
            players= self.create_players_dict()
            data_frame = []
            for element in players:
                data_frame += element['players']
            final_df= pd.DataFrame (data_frame)    
            # final_df= pd.concat(list_dfs).reset_index(drop = True)
            final_df['nationality']=final_df['nationality'].apply(lambda x: x[0])
            final_df= final_df.drop_duplicates(ignore_index= True)
            self.logger.info ('DataFrame created')
            return final_df
        except requests.RequestException as e:
            self.logger.error (f'Players DataFrame creation failed. Error: {e}')
            return None
        
    
    def transfers_df (self, create= True, path= None):
        try:
            if create == True:
                players= self.create_players_df()
            else:
                players= pd.read_csv(path)    
            players= list(players['id'])
            transfers_urls= [f'{self.url}players/{i}/transfers'for i in players]
            transfers_list=[]
            for i in transfers_urls:
                response= requests.get(i)
                data= response.json()
                transfers_list.append(data)
            list_dfs=[]    
            for index, element in enumerate(transfers_list):
                df= pd.DataFrame(transfers_list[index]['transfers'])
                df['id']= transfers_list[index]['id']
                df['from']= df['from'].apply(lambda x: x['clubID'])
                df['to']= df['to'].apply(lambda x: x['clubID'])
                list_dfs.append(df)
            df= pd.concat(list_dfs)
            self.logger.info ('Transfers DataFrame created')
            return df
        except requests.RequestException as e:
            self.logger.error (f'Transfers DataFrame creation failed. Error: {e}')
            return None
        
    def profile_df  (self, create= True, path= None):
        try:    
            if create == True:
                players= self.create_players_df()
            else:
                players= pd.read_csv(path) 
            players= list(players['id'])
            profile_urls= [f'{self.url}players/{i}/profile'for i in players]
            profile_list= []
            for i in profile_urls:
                response= requests.get(i)
                data= response.json()
                profile_list.append(data)
            list_dfs= []
            
            for index, element in enumerate(profile_list):
                key_columns= ['id','age','citizenship', 'position']
                profile_list[index]={k:profile_list[index][k] for k in key_columns} 
                if isinstance(profile_list[index]['citizenship'], list):
                    profile_list[index]['citizenship']= profile_list[index]['citizenship'][0]
                if isinstance(profile_list[index]['position'], dict):
                    profile_list[index]['position']= profile_list[index]['position']['main']
                df= pd.DataFrame([profile_list[index]])
                list_dfs.append(df)
            df= pd.concat(list_dfs)
            self.logger.info ('Profiles DataFrame created')
            return df
        except requests.RequestException as e:
            self.logger.error (f'Profile DataFrame creation failed. Error: {e}')
            return None
        
    def stats_df  (self, create= True, path= None):
        try:
            if create == True:
                players= self.create_players_df()
            else:
                players= pd.read_csv(path) 
            players= list(players['id'])
            stats_urls= [f'{self.url}players/{i}/stats'for i in players]
            stats_list= []
            for i in stats_urls:
                response= requests.get(i)
                data= response.json()
                stats_list.append(data)
            list_dfs= []
            pattern = r"(\d{2})/\d{2}"
            def replace_year(match):
                return "20" + match.group(1)
            def transform_season_to_year(season):
                return re.sub(pattern, replace_year, season)
            for index, element in enumerate(stats_list):
                
                    df= pd.DataFrame(stats_list[index]['stats'])
                    df['id']= stats_list[index]['id']
                    if 'seasonID' in df.columns:
                         df['seasonID'] = df['seasonID'].apply(transform_season_to_year)
                    list_dfs.append(df)
            
            df= pd.concat(list_dfs)
            self.logger.info ('Stats DataFrame created')
            return df
        except requests.RequestException as e:
            self.logger.error (f'Stats DataFrame creation failed. Error: {e}')
            return None

    def players_df (self):
        pass

        








 




    

    
  