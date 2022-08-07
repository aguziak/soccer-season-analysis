import requests
import pandas as pd
import os
import json
import io
import colorsys
import urllib.request
import csv

from PIL import Image


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


def get_season_data(season: int, league_id: int, use_cache=False) -> pd.DataFrame:
    """
    Gets the historical data for a given season and soccer league

    Args:
        season (int): Season for which to get data
        league_id (str): League for which to get data
        use_cache (bool): Whether to use local cache or not

    Returns:
        DataFrame: Historical data for the league in the given season

    """
    cache_dir = os.path.abspath(f'../local_cache/')
    cache_file_path = f'{cache_dir}/league_{league_id}_season_{season}.csv'

    if use_cache:
        print('Retrieving data from local cache')
        file_already_exists = os.path.exists(cache_file_path)

        if file_already_exists:
            return pd.read_csv(cache_file_path)

    print('Retrieving data from API')
    api_key = _get_rapid_api_key()

    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

    querystring = {"league": league_id, "season": season}

    headers = {
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
        'x-rapidapi-key': api_key
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    df = pd.DataFrame(json.loads(response.text)['response'])

    df = _flatten_response(df)

    if use_cache:
        print('Saving data to local cache')
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(cache_file_path)

    return df


def get_team_colors_from_api(matches_df: pd.DataFrame, use_cache=False, league_id=None, season=None,
                             rebuild_cache=False):
    """
    Gets a dictionary of team colors by finding the most common pixel color in the teams' logo

    Requests a 256 x 256 PNG image for each team using the API-Football API

    Args:
        matches_df (DataFrame): DataFrame acquired using the get_season_data function
        use_cache (bool): Optional parameter specifying whether to use cached data if it is available. Cache
            is granular down to the league/season combination. Must specify league_id and season to use the
            cache
        league_id (int): League id for the team colors to retrieve
        season (int): Season for the team colors to retrieve
        rebuild_cache (bool): If true, will recreate data using the API and overwrite any existing cached data with
            the new values

    Returns:
        dict: Dictionary where the keys are team ids as ints and the values are hex colors
    """
    cache_dir = os.path.abspath(f'../local_cache/')
    cache_file_path = f'{cache_dir}/league_{league_id}_season_{season}_color_data.csv'

    if use_cache and not rebuild_cache:
        if league_id is None or season is None:
            raise RuntimeError('One of league_id or season is None when attempting to use colors cache!')

        file_already_exists = os.path.exists(cache_file_path)

        if file_already_exists:
            print('Retrieving color data from local cache')
            return pd.read_csv(cache_file_path)

    def get_nth_most_common_color(colors: list, n: int):
        """
        Finds and returns the nth most common color in a list of colors in HLS format

        Args:
            colors: List of colors in RGB format
            n: Rank of the color to get ordered by descending frequency

        Returns:
            Tuple: Tuple containing hue, saturation, lightness in that order
        """
        colors.sort(key=lambda color: color[0])
        most_common_color = colors[-n][-1]
        most_common_color_hls = colorsys.rgb_to_hsv(most_common_color[0], most_common_color[1],
                                                    most_common_color[2])
        return most_common_color_hls[0], most_common_color_hls[1], most_common_color_hls[2]

    def get_colors(df: pd.DataFrame):
        """
        Finds the most common color in a team's logo. Uses HSV color system to filter out low saturation colors
            that are likely to be background pixels. Doesn't always return the best color to represent a team
            but is usually pretty decent. Best used as a fallback option when a hand-picked value isn't available.

        Args:
            df (DataFrame): DataFrame containing at least a teams-home-logo column with URLs pointing to team logo
                image files served by API-FOOTBALL in Rapid API and a teams-home-id containing a team id.

        Returns:
            DataFrame: DataFrame indexed by team id with a column named team-color containing the hex value
                of the most common pixel found in the team's logo.
        """
        logo_url = df.iloc[0]['teams-home-logo']
        with urllib.request.urlopen(logo_url) as image_url_in:
            image_file = io.BytesIO(image_url_in.read())
        image = Image.open(image_file)
        image: Image

        image = image.convert('RGB')

        image_colors = image.getcolors(image.size[0] * image.size[1])
        image_colors.sort(key=lambda color: color[0])

        current_rank = 1
        h, s, v = get_nth_most_common_color(image_colors, current_rank)
        while s < 0.35:
            current_rank += 1
            h, s, v = get_nth_most_common_color(image_colors, current_rank)

        rgb_color_to_use = colorsys.hsv_to_rgb(h, s, v)
        return '#%02x%02x%02x' % \
               (round(rgb_color_to_use[0]), round(rgb_color_to_use[1]), round(rgb_color_to_use[2]))

    colors_df = matches_df.groupby(by='teams-home-id') \
        .apply(get_colors) \
        .rename({0: 'team-color'}, axis=1)

    if use_cache or rebuild_cache:
        print('Saving color data to local cache')
        os.makedirs(cache_dir, exist_ok=True)
        colors_df.to_csv(cache_file_path)

    return colors_df


if __name__ == '__main__':
    epl_league_id = 39
    result = get_season_data(2018, epl_league_id)
    print(result.head())
