import requests
import pandas as pd
import os


def _get_rapid_api_key() -> str:
    with open(os.path.abspath('../api_keys/rapid_api_key.txt')) as file:
        api_key = file.read()
    return api_key


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
    return pd.DataFrame()


if __name__ == '__main__':
    result = get_season_data(2020, 'epl')
