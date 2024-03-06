import pandas as pd

class Player:
    
    def __init__(self, player_id, player_dict, player_name):
        self.player_id = player_id
        self.player_dict = player_dict
        self.player_name = player_name
    
    def player_data_real(self):
        try:
            # Próba pobrania danych
            df_list = pd.read_html(f'https://fbref.com/en/players/{self.player_id}/{self.player_name}')
            df_list = df_list[-9:]  # Ostatnie 9 DataFrame'ów
            keys = list(self.player_dict.keys())

            # Upewnienie się, że mamy odpowiednią liczbę DataFrame'ów
            if len(keys) != len(df_list):
                raise ValueError("Liczba kluczy i DataFrame'ów nie jest równa.")

            # Tworzenie słownika DataFrame'ów
            dict_dfs = {keys[i]: df for i, df in enumerate(df_list)}

            # Przypisywanie MultiIndex do każdego DataFrame'u
            for key in keys:
                if key in dict_dfs and len(self.player_dict[key]) == len(dict_dfs[key].columns):
                    player_dict_tup = [tuple(col) for col in self.player_dict[key]]
                    multi_index = pd.MultiIndex.from_tuples(player_dict_tup)
                    dict_dfs[key].columns = multi_index
                else:
                    raise ValueError(f"Nie można przypisać MultiIndex do DataFrame'u dla klucza '{key}'.")

            return dict_dfs

        except Exception as e:
            # Obsługa wyjątków, np. problemów z połączeniem internetowym, błędów w danych itp.
            print(f"Wystąpił błąd: {e}")
            return None
        
        def player_data_fifa(self, version):
            pass
        def player_data_fm (self):
            pass


