
import requests
from DataSource import  Players, Clubs
import logging, re
import time

#England
instance_eng= Players('GB1', '2022','2023')
path= 'Data/Input/England/main_players_df_check.csv'
data_england= instance_eng.create_players_df()
data_england.to_csv(path)
time.sleep(60)
data_transfers_england= instance_eng.transfers_df(create=False, path= path)
data_transfers_england.to_csv('Data/Input/England/transfers_df.csv')
time.sleep(60)
data_stats_england= instance_eng.stats_df(create=False, path=path)
data_stats_england.to_csv('Data/Input/England/stats_df.csv')

#Spain
instance_es= Players('ES1', '2022','2023')
path= 'Data/Input/Spain/main_players_df.csv'
data_es= instance_es.create_players_df()
data_es.to_csv(path)
time.sleep(60)
data_transfers_es= instance_es.transfers_df(create=False, path= path)
data_transfers_es.to_csv('Data/Input/Spain/transfers_df.csv')
time.sleep(60)
data_stats_es= instance_es.stats_df(create=False, path=path)
data_stats_es.to_csv('/Data/Input/Spain/stats_df.csv')

#Italy
instance_it= Players('IT1', '2022','2023')
path= 'Data/Input/Italy/main_players_df.csv'
data_it= instance_it.create_players_df()
data_it.to_csv(path)
time.sleep(60)
data_transfers_it= instance_it.transfers_df(create=False, path= path)
data_transfers_it.to_csv('Data/Input/Italy/transfers_df.csv')
time.sleep(60)
data_stats_it= instance_it.stats_df(create=False, path=path)
data_stats_it.to_csv('Data/Input/Italy/stats_df.csv')

#Germany
instance_de= Players('L1', '2022','2023')
path= 'Data/Input/Germany/main_players_df.csv'
data_de= instance_de.create_players_df()
data_de.to_csv(path)
time.sleep(60)
data_transfers_de= instance_de.transfers_df(create=False, path= path)
data_transfers_de.to_csv('Data/Input/Germany/transfers_df.csv')
time.sleep(60)
data_stats_de= instance_de.stats_df(create=False, path=path)
data_stats_de.to_csv('Data/Input/Germany/stats_df.csv')

#France
instance_fr= Players('FR1', '2022','2023')
path= 'Data/Input/France/main_players_df.csv'
data_fr= instance_fr.create_players_df()
data_fr.to_csv(path)
time.sleep(60)
data_transfers_fr= instance_fr.transfers_df(create=False, path= path)
data_transfers_fr.to_csv('Data/Input/France/transfers_df.csv')
time.sleep(60)
data_stats_fr= instance_fr.stats_df(create=False, path=path)
data_stats_fr.to_csv('Data/Input/France/stats_df.csv')


