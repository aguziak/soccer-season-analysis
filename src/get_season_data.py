import requests
import pandas as pd


def get_season_data(season: int, league: str) -> pd.DataFrame:
    """
    Gets the historical data for a given season and soccer league

    Args:
        season (int): Season for which to get data
        league (str): League for which to get data

    Returns:
        DataFrame: Historical data for the league in the given season

    """
    return pd.DataFrame()


if __name__ == '__main__':
    result = get_season_data(2020, 'epl')
