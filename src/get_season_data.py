import requests
import pandas as pd
import os
import json


def _get_rapid_api_key() -> str:
    with open(os.path.abspath('../api_keys/rapid_api_key.txt')) as file:
        api_key = file.read()
    return api_key


def _flatten_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with dictionary values and "unrolls" them such that each dictionary entry receives its own column.

    Args:
        df (DataFrame): DataFrame to flatten

    Returns:
        DataFrame: Flat DataFrame

    """
    df_types: pd.Series = df.iloc[0].apply(type)
    df_types = df_types.reset_index().rename({'index': 'column_name', 0: 'column_type'}, axis=1)
    dict_columns = df_types[df_types['column_type'] == dict]['column_name']
    if len(dict_columns) > 0:
        for col_name in dict_columns:
            new_cols = df[col_name].apply(pd.Series)
            new_col_names = {new_col_name: f'{col_name}-{new_col_name}' for new_col_name in new_cols.columns}
            new_cols = new_cols.rename(new_col_names, axis=1)
            df = pd.concat([df, new_cols], axis=1)
            df = df.drop(col_name, axis=1)
        df = _flatten_response(df)

    return df


def get_season_data(season: int, league: str) -> pd.DataFrame:
    """
    Gets the historical data for a given season and soccer league

    Args:
        season (int): Season for which to get data
        league (str): League for which to get data

    Returns:
        DataFrame: Historical data for the league in the given season

    """
    api_key = _get_rapid_api_key()

    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

    querystring = {"league": "39", "season": "2020", "from": "2021-01-01", "to": "2021-04-07"}

    headers = {
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
        'x-rapidapi-key': api_key
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    df = pd.DataFrame(json.loads(response.text)['response'])

    df = _flatten_response(df)

    return df


if __name__ == '__main__':
    result = get_season_data(2020, 'epl')
