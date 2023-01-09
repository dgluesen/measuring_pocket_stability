import numpy as np
import pandas as pd
import datetime

from src.config import OLINE_PFF_ROLE, DLINE_PFF_ROLE, QB_PFF_ROLE


def euclidean_distance(p0: np.array, p1: np.array) -> float:
    """Computes the euclidean distance between two points."""
    return np.sqrt(np.sum((p0 - p1) ** 2))


def min_distance_to_point(p: np.array, p_list: list) -> float:
    """Extracts the minimal distance from a list of points to a given point."""
    distances = [euclidean_distance(p0=p, p1=np.array(x)) for x in p_list]
    return np.min(distances)


def pocket_collapse(qb: np.array, oline: list, dline: list) -> bool:
    """Evaluates whether the pocket has collapsed or not."""
    oline_dist = min_distance_to_point(qb, oline)
    dline_dist = min_distance_to_point(qb, dline)
    return dline_dist < oline_dist


def pocket_collapse_frame(df_frame: pd.DataFrame) -> bool:
    """Evaluates per frame whether the pocket has collapsed or not."""
    oline = df_frame[df_frame.pff_role == OLINE_PFF_ROLE]["coordinates"].tolist()
    dline = df_frame[df_frame.pff_role == DLINE_PFF_ROLE]["coordinates"].tolist()
    qb = df_frame.loc[df_frame.pff_role == QB_PFF_ROLE, "coordinates"].item()
    return pocket_collapse(qb=qb, oline=oline, dline=dline)


def time_play(df_play: pd.DataFrame) -> float:
    """Returns the duration of the play."""
    play_start = datetime.datetime.strptime(str(np.min(df_play.time)), "%Y-%m-%dT%H:%M:%S.%f")
    play_end = datetime.datetime.strptime(str(np.max(df_play.time)), "%Y-%m-%dT%H:%M:%S.%f")
    return (play_end - play_start).total_seconds()


def time_pocket(df_play: pd.DataFrame) -> float:
    """Returns the duration the pocket stands in a play."""
    pocket_start = datetime.datetime.strptime(str(np.min(df_play.time)), "%Y-%m-%dT%H:%M:%S.%f")
    if any(df_play.pocket_collapsed):
        tmp = np.min(df_play[df_play.pocket_collapsed].time)
        pocket_end = datetime.datetime.strptime(str(tmp), "%Y-%m-%dT%H:%M:%S.%f")
    else:
        pocket_end = datetime.datetime.strptime(str(np.max(df_play.time)), "%Y-%m-%dT%H:%M:%S.%f")
    return (pocket_end - pocket_start).total_seconds()


def pocket_collapse_play(df_play: pd.DataFrame) -> bool:
    """Evaluates whether the pocket has collapsed in a play or not."""
    return any(df_play.pocket_collapsed)


def evaluate_pocket_stability(data_pocket_collapsed: pd.DataFrame):
    df = pd.DataFrame()
    df["team"] = pd.concat([data_pocket_collapsed.oline, data_pocket_collapsed.dline]).unique()
    df["oline_ps_perc"] = df.apply(
        lambda x: 100 * (
            (sum(np.logical_not(data_pocket_collapsed[data_pocket_collapsed.oline == x.team].pocket_collapsed)) /
             len(data_pocket_collapsed[data_pocket_collapsed.oline == x.team].pocket_collapsed))),
        axis=1).round(4)
    df["oline_above_avg"] = df["oline_ps_perc"] >= df["oline_ps_perc"].mean()
    df["dline_pb_perc"] = df.apply(
        lambda x: 100 * ((sum(data_pocket_collapsed[data_pocket_collapsed.dline == x.team].pocket_collapsed) /
                          len(data_pocket_collapsed[data_pocket_collapsed.dline == x.team].pocket_collapsed))),
        axis=1).round(4)
    df["dline_above_avg"] = df["dline_pb_perc"] >= df["dline_pb_perc"].mean()
    return df
