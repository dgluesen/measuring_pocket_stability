import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pocket_stability_measures import pocket_collapse
from src.config import OLINE_PFF_ROLE, DLINE_PFF_ROLE, QB_PFF_ROLE, RELEVANT_ROLE, PATH_WEEKS_LIST, PATH_PFF


def visualize_pocket(df_play: pd.DataFrame, frames: list) -> None:
    """Visualizes the pocket in a play for given frames"""
    plot_list = []
    fig, ax = plt.subplots(1, len(frames), figsize=(12, 5), sharex=True, sharey=True)
    for i, frame in enumerate(frames):
        df_frame = df_play[df_play.frameId == frame]
        if len(df_frame) == 0:
            break
        oline = df_frame[df_frame.pff_role == OLINE_PFF_ROLE]["coordinates"].tolist()
        dline = df_frame[df_frame.pff_role == DLINE_PFF_ROLE]["coordinates"].tolist()
        qb = df_frame.loc[df_frame.pff_role == QB_PFF_ROLE, "coordinates"].item()
        collapse = pocket_collapse(qb=qb, oline=oline, dline=dline)
        x_oline, y_oline = zip(*oline)
        x_dline, y_dline = zip(*dline)
        # ax[i].scatter(qb[0], qb[1], c="red", alpha=.3)
        # ax[i].scatter(x_oline, y_oline, c="blue", alpha=.3)
        # ax[i].scatter(x_dline, y_dline, c="green", alpha=.3)
        # ax[i].legend()
        # ax[i].set_title("Frame = " + str(frame) + " - Pocket collapsed = " + str(collapse))

        plot_list.append(ax[i].scatter(qb[0], qb[1], c="red", alpha=.65, label="QB"))
        plot_list.append(ax[i].scatter(x_oline, y_oline, c="blue", alpha=.65, label="O-line"))
        plot_list.append(ax[i].scatter(x_dline, y_dline, c="green", alpha=.65, label="D-line"))
        ax[i].set_title("Frame = " + str(frame) + "; collapsed = " + str(collapse), fontsize=10)
        ax[i].set_xlim(30, 45)
        ax[i].set_ylim(18.5, 33.5)
        ax[i].grid(True)
        ax[i].set_axisbelow(True)

    fig.suptitle("Structure of the pocket until collapse", fontsize=14)
    fig.legend(handles=plot_list[0:3], loc="upper right")

def get_pocket_visualization_data(week_index: int, game: int, play: int) -> pd.DataFrame:
    """Gets example data for visualization."""
    data_week = pd.read_csv(PATH_WEEKS_LIST[week_index],
                            usecols=["gameId", "playId", "nflId", "frameId", "time", "x", "y"])
    data_week = data_week[(data_week.gameId == game) & (data_week.playId == play)]
    if len(data_week) == 0:
        print("no example data for this combination of week, game, and play!")
        return None
    data_pff = pd.read_csv(PATH_PFF, usecols=["gameId", "playId", "nflId", "pff_role"]
                           ).query(f"pff_role in {RELEVANT_ROLE}")
    data_raw = pd.merge(data_week, data_pff, on=["gameId", "playId", "nflId"])
    data_raw["coordinates"] = data_raw.apply(lambda row: np.array([row.x, row.y]), axis=1)
    return data_raw


def visualize_pocket_frame(df_frame: pd.DataFrame) -> None:
    """Visualizes the pocket in a play for given frames"""
    oline = df_frame[df_frame.pff_role == OLINE_PFF_ROLE]["coordinates"].tolist()
    dline = df_frame[df_frame.pff_role == DLINE_PFF_ROLE]["coordinates"].tolist()
    qb = df_frame.loc[df_frame.pff_role == QB_PFF_ROLE, "coordinates"].item()
    collapse = pocket_collapse(qb=qb, oline=oline, dline=dline)
    x_oline, y_oline = zip(*oline)
    x_dline, y_dline = zip(*dline)
    fig, ax = plt.subplots()
    ax.scatter(qb[0], qb[1], c="red", alpha=.3, label="QB")
    ax.scatter(x_oline, y_oline, c="blue", alpha=.3, label="O-Line")
    ax.scatter(x_dline, y_dline, c="green", alpha=.3, label="D-Line")
    ax.legend()
    ax.set_title("Pocket collapsed = " + str(collapse))
