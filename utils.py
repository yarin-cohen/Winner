import pandas as pd
import numpy as np
import datetime
import copy
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

def create_games_df(save_dict):
    games_list_dict = []

    for game_id in save_dict['games']:
        game_desc_dict = copy.deepcopy(save_dict['games'][game_id])
        game_desc_dict['game_id'] = int(game_id)
        ids_home = []
        home_table = game_desc_dict['home_table']
        for k in range(len(home_table)):
            if not np.isnan(home_table[k]['#']):
                ids_home.append(home_table[k]['player_id'])

        ids_away = []
        away_table = game_desc_dict['away_table']
        for k in range(len(away_table)):
            if not np.isnan(away_table[k]['#']):
                ids_away.append(away_table[k]['player_id'])

        # padding with zeros up to 13 team players:
        diff_home = 13 - len(ids_home)
        diff_away = 13 - len(ids_away)
        if diff_home > 0:
            for k in range(diff_home):
                ids_home.append('0')
        if diff_away > 0:
            for k in range(diff_away):
                ids_away.append('0')

        # adding new keys for each player id
        for k in range(len(ids_home)):
            game_desc_dict['player_home' + str(k)] = int(ids_home[k])
        for k in range(len(ids_away)):
            game_desc_dict['player_away' + str(k)] = int(ids_away[k])

        game_desc_dict.pop('home_table', None)
        game_desc_dict.pop('away_table', None)

        games_list_dict.append(game_desc_dict)

    games_df = pd.DataFrame(games_list_dict)

    # adding month idx column
    games_dates = games_df['datr']

    base_date = datetime.datetime(1990, 9, 23)
    month_idx = []
    for k in range(len(games_dates)):
        month_idx.append((games_dates[k].year - base_date.year) * 12 + games_dates[k].month - base_date.month)

    games_df['month_idx'] = month_idx
    return games_df


def get_mean_stats(col_name, lines):
    no_nan_lines = lines[np.logical_not(np.isnan(lines[col_name]))]
    if len(no_nan_lines) == 0:
        return 0, 0

    no_nan_lines[col_name] = no_nan_lines[col_name].astype(float)
    player_mean_stats = no_nan_lines.groupby('player_id')[col_name].mean()[0]
    return player_mean_stats, len(no_nan_lines)


def create_history_obj(games_df, game_line, players_df, player_id):
    """games dict is a dictionary. the keys contain the game ids so each game stats can be accessed by game id. the value of the
    dictionary for each key is a list of player stats. each element of the list contains a dataframe of player lines from games
    previos to the current game including the current game

    {game_id---> [player1_lines_of_stats_from_prev_games, player2_lines_of_stats_from_prev_games, ...]}
    this dictionary is saved in games_dict_for_neural.pkl"""

    game_date = game_line['datr']
    player_relevant_lines = players_df[players_df['player_id'] == str(player_id)]
    player_relevant_lines = player_relevant_lines[np.logical_and(player_relevant_lines['datr'] <= game_date,
                                                                 player_relevant_lines['datr'] > game_date +
                                                                 relativedelta(months=-12))]
    player_relevant_lines = player_relevant_lines.drop('game_name', axis=1)
    player_relevant_lines = player_relevant_lines.drop('player_name', axis=1)
    player_relevant_lines = player_relevant_lines.drop('team_name', axis=1)
    player_relevant_lines = player_relevant_lines.drop('#', axis=1)
    player_relevant_lines['is_player_home'] = np.zeros((len(player_relevant_lines), 1)).tolist()
    relevant_columns = player_relevant_lines.columns
    bad_idxs = []
    for index, player_line in player_relevant_lines.iterrows():
        cur_game_id = player_line['game_id']
        mask = (games_df['game_id'].astype(str) == cur_game_id)
        cur_game = games_df[mask]
        if len(cur_game) == 0:
            bad_idxs.append(index)
        team_id = player_line['team_id']

        for rel_col in relevant_columns:
            stat = player_line[rel_col]
            if rel_col == 'datr' or 'id' in rel_col:
                continue
            if np.isnan(stat):
                player_relevant_lines.set_value(index, rel_col + '_is_valid', 0)
                player_relevant_lines.set_value(index, rel_col, 0)
            else:
                player_relevant_lines.set_value(index, rel_col + '_is_valid', 1)
        if len(cur_game) == 0:
            player_relevant_lines.set_value(index, 'is_player_home', 0)
            player_relevant_lines.set_value(index, 'is_player_home_is_valid', 0)
        else:
            is_player_home = int(str(team_id) == cur_game['home_team_id'])
            player_relevant_lines.set_value(index, 'is_player_home', is_player_home)

    return player_relevant_lines


# preprocess by max min
def preprocess(total_mat):
    saved_cols = []
    max_cols = []
    min_cols = []

    for k in tqdm(range(55)):
        spec_col = total_mat[:, :, k, :]
        if len(np.unique(spec_col)) <= 2:  # only 0, 1
            continue
        else:
            max_col = np.max(spec_col)
            min_col = np.min(spec_col)
            max_cols.append(max_col)
            min_cols.append(min_col)

            if not (max_col - min_col) == 0:
                total_mat[:, :, k, :] /= (max_col - min_col)
                saved_cols.append(k)

    return total_mat, saved_cols, max_cols, min_cols


def preprocess_by_stats(total_mat, saved_cols, max_cols, min_cols):
    for k in tqdm(range(len(saved_cols))):
        if not (max_cols[k] - min_cols[k]) == 0:
            total_mat[:, :, saved_cols[k], :] /= (max_cols[k] - min_cols[k])

    return total_mat
