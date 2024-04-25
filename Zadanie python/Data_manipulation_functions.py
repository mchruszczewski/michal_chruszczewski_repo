import pandas as pd 
import numpy as np
import os 
from pathlib import Path
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



def join_tables(tables_list):
    """
    Joins multiple DataFrames stored in a list into a single DataFrame based on a common key, with index resetting.
    This function performs a sequential left join operation on the DataFrames listed in `tables_list` using a specified column ('id') as the merge key.

    Parameters:
    - tables_list (list of pd.DataFrame): A list containing at least three DataFrames to be joined. The DataFrames
      in the list are expected to contain a common column 'id' which is used as the key for joining.

    Returns:
    - pd.DataFrame: A DataFrame resulting from the left joins of the DataFrames in `tables_list`. The index of the resulting DataFrame is reset to ensure continuity and uniqueness.

    Example:
    - Assuming three DataFrames df1, df2, df3 in `tables_list` with a common key 'id':
      result_df = join_tables([df1, df2, df3])

    Note:
    - This function assumes the DataFrames are in a specific order within the list and all contain the column 'id'.
      If these conditions are not met, the function may result in errors or incorrect joins.
    """
    # Merge the second and third DataFrame in the list and reset the index
    df = tables_list[1].merge(tables_list[2], on='id', how='outer').reset_index(drop=True)
    # Merge the resulting DataFrame with the first DataFrame in the list and reset the index
    df = df.merge(tables_list[0], on='id', how='outer').reset_index(drop=True)
    return df


def concat_tables(countries_list):
    """
    Concatenates a list of DataFrames into a single DataFrame and resets the index to provide a unique and continuous sequence.
    This function stacks the DataFrames vertically (i.e., along axis=0), appending each DataFrame below the previous one.

    Parameters:
    - countries_list (list of pd.DataFrame): A list of DataFrames to be concatenated.

    Returns:
    - pd.DataFrame: A single DataFrame created by vertically concatenating all the DataFrames from the `countries_list`.
      The resulting DataFrame's index is reset to ensure it is unique and continuous.

    Example:
    - Assuming two or more DataFrames df1, df2,..., dfn in `countries_list`:
      result_df = concat_tables([df1, df2, ..., dfn])

    Note:
    - This function concatenates DataFrames vertically and resets the index to avoid duplicate indices.
      Ensure that the DataFrames are properly aligned or compatible for such operations to avoid data misalignment.
    """
    return pd.concat(countries_list).reset_index(drop=True)




def retrieve_directories(path):
    """
    Lists all directories within a specified directory using pathlib.

    Parameters:
    - path (str): The path to the directory where to list subdirectories.

    Returns:
    - List of directories found within the specified path.
    """
    directory_path = Path(path)

    if not directory_path.is_dir():
        return "Provided path is not a directory or does not exist."

    directories = [d.name for d in directory_path.iterdir() ]
    if 'Other' in directories:
        directories.remove('Other')
    
    return directories

def adding_competitions(df, ref_data):
    ref_data= pd.read_csv(ref_data)
    competitions_dict= {'GB1': 'England', 'FR1': 'France', 'ES1':'Spain', 'IT1': 'Italy', 'L1':'Germany'}
    df= df.merge(ref_data, how= 'left', left_on= 'from', right_on= 'id', suffixes= ('_main', '_comp'))
    df['Competitions']= df['Competitions'].map(competitions_dict)
    df['Competitions']= df['Competitions'].fillna('Other')
    df = df.loc[:, ~df.columns.str.contains('_comp')]
    return df


def raw_df(path):
    """
    Processes CSV files within subdirectories of a given directory, each subdirectory representing a country.
    This function assumes each country directory contains further subdirectories with CSV files that need to be
    read, merged, and then all country results are concatenated into a final DataFrame.

    Parameters:
    - path (str): The path to the main directory containing subdirectories named after countries. Each country
      directory is expected to contain further directories with CSV files.

    Returns:
    - pd.DataFrame: A DataFrame containing merged data from all the CSV files across all country-specific
      directories. The DataFrames are joined based on common columns identified in `join_tables` and then
      concatenated into a final DataFrame.

    Workflow:
    1. Retrieve the names of subdirectories within the main directory, each representing a country.
    2. For each country, retrieve further subdirectory paths containing CSV files.
    3. Load all CSV files from these subdirectories into DataFrames.
    4. Merge these DataFrames using the `join_tables` function.
    5. Append the result to a list of DataFrames.
    6. Concatenate all country-specific merged DataFrames into a final DataFrame using `concat_tables`.

    Example Usage:
    - Assuming the directory '/data/countries/' contains multiple country directories each with subdirectories
      of CSV files:
      final_df = tables_list('/data/countries/')

    Notes:
    - This function relies on 'retrieve_directories' to list directories, 'join_tables' to merge DataFrames within
      each country based on specific logic, and 'concat_tables' to concatenate the results.
    - It is crucial that CSV files within subdirectories are compatible for merging, i.e., they must contain the
      necessary columns specified in the merging logic of `join_tables`.
    """
    list_countries = retrieve_directories(path)
    list_tables = []

    for i in list_countries:
        path_country = os.path.join(path, i)
        # Retrieve paths to subdirectories containing CSV files within each country directory
        directories_names = [os.path.join(path_country, d) for d in retrieve_directories(path_country)]
        # Read each CSV file into a DataFrame
        tables_list = [pd.read_csv(table) for table in directories_names]
        # Merge these DataFrames
        df = join_tables(tables_list)
        # Append the merged DataFrame to the list of DataFrames
        list_tables.append(df)

    # Concatenate all merged DataFrames into one final DataFrame
    final_df = concat_tables(list_tables)

    path_clubs_md= os.path.join(path,'Other','clubs_md.csv')
    final_df= adding_competitions(final_df,path_clubs_md)

    return final_df
    

def delete_strings_cash(row):
    """
    Processes a given string or other data type, applying specific transformations
    based on its content, primarily for cleaning or adjusting financial data strings.

    The function performs the following operations on a string input:
    1. Converts certain strings to NaN:
       - Strings containing 'loan', 'fee', '?', 'draft', or 'free' are deemed irrelevant
         or uninformative and are replaced with NaN (missing value indicator in pandas).
    2. Converts strings ending with 'm' or 'k' and containing monetary values in Euros
       to a numerical value (float):
       - Strings ending with 'm' are treated as millions. Any 'm' and '€' characters
         are removed, and the number is converted to float and multiplied by 1e6.
       - Strings ending with 'k' are treated as thousands. Any 'k' and '€' characters
         are removed, and the number is converted to float and multiplied by 1e3.

    Parameters:
    - row : Any data type, but intended for use with strings. The function directly returns
            any input that is not a string without modification.

    Returns:
    - The transformed data, which could be a float if the input string is a financial value
      ending in 'm' or 'k', np.nan for specific unwanted string patterns, or the input itself
      if it does not meet any criteria for modification.

    Example:
    >>> delete_strings('100m€')
    100000000.0
    >>> delete_strings('fee')
    nan
    >>> delete_strings(150)
    150
    """
    if isinstance(row, str):
        if ('loan' in row) or ('fee' in row) or ('?' in row) or ('draft' in row) or ('free' in row):
            row= np.nan
            return row
        elif row.endswith('m'):
            row= row.replace('m', '')
            row= row.replace('€', '')
            row = float(row)*1e6
            return row
        elif row.endswith('k'):
            row= row.replace('k', '')
            row= row.replace('€', '')
            row = float(row)*1e3
        elif row.startswith('€'):
            row =row.replace('€', '')
            return float(row)

    return row

def delete_strings_date(row):
    """
    Extracts and converts the last four characters of a string to a float.
    Intended for use with date strings where only the year part is needed.
    
    Parameters:
    - row (str or other): The input data which can be a string or any other type.
    
    Returns:
    - float: The year extracted from the string, converted to float. If input is not
      a string, returns the input as it is.
    """
    if isinstance(row, str):
        row = row[-4:]
    return float(row)

def delete_strings_season(row):
    """
    Processes a string representing a season by extracting the first two characters,
    appending '20' to make a complete year, and returning it.
    
    Parameters:
    - row (str or other): The input data which can be a string or any other type.
    
    Returns:
    - str or original type: Transformed year string if input was a string,
      otherwise returns the input unchanged.
    """
    if isinstance(row, str):
        row = row[0:2]
        row = '20' + row
        return row
    return row

def positions_clusters(row):
    # Check if the string contains 'Back'
    if 'Back' in row:
        return 'Defender'
    # Check if the string contains 'Midfield'
    elif 'Midfield' in row:
        return 'Midfielder'
    # Check if the string contains 'Winger'
    elif 'Winger' in row:
        return 'Winger'
    # Check if the string contains 'Striker' or 'Forward'
    elif 'Striker' in row or 'Forward' in row:
        return 'Forward'
    else:
        return 'Goalkeeper'

    
def grouping(df):
    """
    Filters and aggregates a DataFrame based on specified criteria and groupings.
    It filters data where 'seasonID' is less than or equal to 'season', sorts,
    and then drops specific columns before grouping by various player and transfer details.

    Parameters:
    - df (DataFrame): The DataFrame to process.
    
    Returns:
    - DataFrame: The grouped DataFrame with sums of certain numerical columns.
    """
    df = df.loc[df['seasonID'] <= df['season'].astype('float')].sort_values(by=['id', 'season']).reset_index(drop=True)
    df = df.drop(columns=['season', 'seasonID', 'clubID', 'competitionID'])
    df= df.drop_duplicates()
    df = df.groupby(['id', 'from', 'to','Competitions', 'transferDate', 'marketValue', 'fee',
                     'position', 'dateOfBirth', 'age', 'nationality', 'currentClub',
                     'height', 'foot', 'competitionName']).sum(['appearances', 'yellowCards', 'goalsConceded',
                                                               'cleanSheets', 'minutesPlayed', 'redCards', 'goals', 'assists',
                                                               'secondYellowCards']).reset_index()
    return df



def changing_type(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype('category')
        elif df[column].dtype == 'float64':
            # Convert float64 to 'Int64' (pandas' nullable integer type) to handle possible NaN values gracefully
            df[column] = df[column].astype('int')
    return df






def data_cleansing(df):
    """
    Cleanses and prepares the transfer data in a DataFrame by renaming columns, converting data types,
    filtering content, and calculating new metrics. It applies a series of transformations to prepare
    the data for analysis.

    Parameters:
    - df (DataFrame): The DataFrame to be cleansed and transformed.
    
    Returns:
    - DataFrame: The cleansed and transformed DataFrame ready for further analysis.
    """
    columns_list = ['id_main', 'from', 'to','Competitions' ,'date', 'season', 'marketValue_x', 'fee', 'position', 'dateOfBirth', 'age', 'nationality', 'currentClub', 'height', 'foot', 'competitionID', 'clubID', 'seasonID', 'competitionName', 'appearances', 'yellowCards', 'goalsConceded',
                    'cleanSheets', 'redCards', 'goals', 'assists', 'secondYellowCards']
    df = df.loc[:, columns_list]
    df = df.rename(columns={'marketValue_x': 'marketValue', 'date': 'transferDate', 'id_main':'id'})
    df['marketValue'] = df['marketValue'].apply(delete_strings_cash)  # Note: 'delete_strings_cash' needs definition
    df['fee'] = df['fee'].apply(delete_strings_cash)  # Note: 'delete_strings_cash' needs definition
    df['transferDate'] = df['transferDate'].apply(delete_strings_date)
    df['dateOfBirth'] = df['dateOfBirth'].apply(delete_strings_date)
    df['season'] = df['season'].apply(delete_strings_season)
    df.loc[~df['competitionName'].isin(['Premier League', 'Serie A', 'LaLiga', 'Ligue 1', 'Bundesliga', 'Champions League', 'Europa League', 'Conference League']), 'competitionName'] = 'Other'
    df = df.loc[df.fee.notna()]
    df = grouping(df)
    df = df.fillna(0)
    df['ageTransfer'] = df['transferDate'] - df['dateOfBirth']
    df= df.drop(columns= ['from', 'to', 'dateOfBirth', 'age', 'currentClub'] )
    df['position']= df['position'].apply(positions_clusters)
    df['above_market']= 0
    df.loc[df['fee']>df['marketValue'],'above_market']= 1
    df['fee']= df['fee']/1000000
    df['marketValue']= df['marketValue']/1000000
    df= changing_type(df)

    return df


def splitting_datasets_class (df):
  label_encoder = LabelEncoder()

  for column in df.columns:
      if df[column].dtype == 'category':
          df[column]= label_encoder.fit_transform(df[column])
  X = df.drop( ['fee','above_market','id','cluster'], axis=1)
  y = df['above_market']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

  return X_train, X_test, y_train, y_test

def scalling_datasets (df):
  scaler = StandardScaler()
  df = scaler.fit_transform(df)
  return df


        


