import numpy as np
import pandas as pd

from src.pocket_stability_measures import pocket_collapse_frame, pocket_collapse_play, time_pocket, time_play
from src.config import PATH_PFF, PATH_WEEKS_LIST, PATH_PLAYS, PATH_POCKET_COLLAPSED, RELEVANT_ROLE, PATH_ML_INPUT


def get_pocket_collapsed_data_week(week_index: int) -> pd.DataFrame:
    """Computes whether the pocket is collapsed or all games, plays, and frames in one week."""
    # raw data
    #data_week = pd.read_csv(
    #    PATH_WEEKS_LIST[week_index], usecols=["gameId", "playId", "nflId", "frameId", "time", "x", "y"])
    #data_pff = pd.read_csv(PATH_PFF, usecols=["gameId", "playId", "nflId", "pff_role"]
    #                       ).query(f"pff_role in {RELEVANT_ROLE}")

    ### 2nd effort: including event data for more precise timing ###
    data_week = pd.read_csv(
        PATH_WEEKS_LIST[week_index], usecols=["gameId", "playId", "nflId", "frameId", "time", "x", "y", "event"])

    data_week['qb_hold'] = ['hold' if ((x=='ball_snap') |
                                       (x=='autoevent_ballsnap'))\
                            else 'no_hold' if ((x=='pass_forward') |
                                               (x=='autoevent_passforward') |
                                               (x=='qb_strip_sack') |
                                               (x=='qb_sack') |
                                               (x=='run'))
                            else None\
                            for x in data_week['event']]
    data_week['qb_hold'] = data_week['qb_hold'].ffill()
    data_week = data_week[data_week['qb_hold'] == 'hold'].drop('qb_hold', axis=1)
    data_pff = pd.read_csv(PATH_PFF, usecols=["gameId", "playId", "nflId", "pff_role"]
                           ).query(f"pff_role in {RELEVANT_ROLE}")
    ### end ###

    data_raw = pd.merge(data_week, data_pff, on=["gameId", "playId", "nflId"])
    data_raw["coordinates"] = data_raw.apply(lambda row: np.array([row.x, row.y]), axis=1)

    # handle exceptions (for some reason, there is no QB in this play) - manually for now
    data_raw.drop(data_raw[((data_raw.gameId == 2021101700) & (data_raw.playId == 2372))].index, inplace=True)

    # collapsed pocket per frame
    data_collapse_frame = data_raw[["gameId", "playId", "frameId", "time"]].copy().drop_duplicates()
    data_collapse_frame["pocket_collapsed"] = data_collapse_frame.apply(
        lambda x: pocket_collapse_frame(
            df_frame=data_raw[
                (data_raw.gameId == x.gameId) & (data_raw.playId == x.playId) & (data_raw.frameId == x.frameId)]),
        axis=1)
    return data_collapse_frame


def get_pocket_stability_data_week(week_index: int) -> pd.DataFrame:
    """Computes the pocket stability measure for all games and plays in one week """
    # raw data
    data_possession = pd.read_csv(PATH_PLAYS, usecols=["gameId", "playId", "possessionTeam", "defensiveTeam"])
    data_possession.rename(columns={"possessionTeam": "oline", "defensiveTeam": "dline"}, inplace=True)
    data_collapse_frame = get_pocket_collapsed_data_week(week_index=week_index)

    # collapsed pocket and times per play
    data_collapse_play = data_collapse_frame[["gameId", "playId"]].copy().drop_duplicates()
    data_collapse_play["pocket_collapsed"] = data_collapse_play.apply(
        lambda x: pocket_collapse_play(
            df_play=data_collapse_frame[
                (data_collapse_frame.gameId == x.gameId) & (data_collapse_frame.playId == x.playId)]),
        axis=1)
    data_collapse_play["pocket_time"] = data_collapse_play.apply(
        lambda x: time_pocket(
            df_play=data_collapse_frame[
                (data_collapse_frame.gameId == x.gameId) & (data_collapse_frame.playId == x.playId)]),
        axis=1)
    data_collapse_play["play_time"] = data_collapse_play.apply(
        lambda x: time_play(
            df_play=data_collapse_frame[
                (data_collapse_frame.gameId == x.gameId) & (data_collapse_frame.playId == x.playId)]),
        axis=1)

    # merge offense and defense to the dataset
    data_collapse_play = data_collapse_play.merge(data_possession, on=["gameId", "playId"])
    return data_collapse_play


def get_pocket_stability_data(out: str = PATH_POCKET_COLLAPSED) -> None:
    """Computes the pocket stability measure for all games and plays."""
    df = pd.concat([get_pocket_stability_data_week(week_index=i) for i in range(len(PATH_WEEKS_LIST))])
    df.to_csv(out, index=False)


def prepare_ml_data(out: str = PATH_ML_INPUT) -> None:
    """Prepares data for ML-model"""
    data_pocket_collapsed = pd.read_csv(
        PATH_POCKET_COLLAPSED, usecols=["gameId", "playId", "pocket_time", "oline", "dline"])
    data_plays = pd.read_csv(
        PATH_PLAYS, usecols=["gameId", "playId", "quarter", "down", "yardsToGo", "pff_passCoverage"])
    data = pd.merge(left=data_pocket_collapsed, right=data_plays, on=["gameId", "playId"])
    data.to_csv(out, index=False)
