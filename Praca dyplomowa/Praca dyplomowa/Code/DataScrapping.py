#%%
import pandas as pd
import numpy as np
import time



#%%

#Adding function

def drop_columns (df_list):
    if isinstance (df_list, list):
        for df in df_list:
            try:
                df.columns= df.columns.droplevel(0)
            except:
                pass
    return df_list



def season_table(id_league, season):
 
    name_ = next(iter(id_league))
    id_ = id_league[name_]
    html_name = name_.strip().replace(' ', '-')
    raw_read = pd.read_html(f'https://fbref.com/en/comps/{id_}/{season}/{season}-{html_name}-Stats')

    table= raw_read[:2]
    table= pd.concat(table, axis=1)
    table= table.sort_values(by='Squad')
    table= table.drop(columns=[('Unnamed: 1_level_0', 'Squad'),('Unnamed: 0_level_0', 'Rk')])


    team_stats= raw_read[2::2]
    team_stats= drop_columns(team_stats)
    team_stats= pd.concat(team_stats, axis=1)
    team_stats= team_stats.drop(columns='Squad')

    # opponent_stats= raw_read[3::2]
    # opponent_stats= drop_columns(opponent_stats)
    # opponent_stats_1= [table.drop(columns='Squad') for table in opponent_stats[1:]]
    # opponent_stats= opponent_stats[:1] + opponent_stats_1
    # opponent_stats= pd.concat(opponent_stats, axis=1)
    # opponent_stats= opponent_stats.rename(columns={'Squad':'AgainstTeam'})


    df= pd.concat([table, team_stats], axis=1)
    df = df.loc[:,~df.columns.duplicated()].copy()
    df=df.reset_index(drop=True)

    
    return df

#Adding frames

#%%
id_leagues= [
    {'Premier League': '9'},
    {'La Liga': '12'},
    {'Serie A': '11'},
    {'Bundesliga': '20'},
   { 'Ligue 1': '13'}
]

season_list= ['2017-2018','2018-2019','2019-2020', '2020-2021', '2021-2022','2022-2023','2023-2024']


#Creating tables 
#%%
season_dfs= []


for season in season_list:
    df = pd.DataFrame()
    for league in id_leagues:
        try:
            df_league = season_table(league, season)
            df = pd.concat([df, df_league],ignore_index=True)
            time.sleep(60)
        except Exception as e:
            print(f"Error processing league {league} in season {season}: {e}")
            time.sleep(60)
            pass
    season_dfs.append(df)

#%%

file_name= 'Data/fbref_db.xlsx'
with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
    for df, sheet in zip(season_dfs, season_list):
        df.to_excel(writer, sheet_name=sheet, index=False)


# %%
