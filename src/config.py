from pathlib import Path

# path to raw data sets (inputs from kaggle)
PATH_GAMES = Path("data", "raw", "games.csv")
PATH_PFF = Path("data", "raw", "pffScoutingData.csv")
PATH_PLAYERS = Path("data", "raw", "players.csv")
PATH_PLAYS = Path("data", "raw", "plays.csv")
PATH_WEEKS_LIST = [Path("data", "raw", "week" + str(i) + ".csv") for i in range(1, 9)]

# path to processed data sets (outputs)
PATH_POCKET_COLLAPSED = Path("data", "processed", "pocket_collapsed.csv")
PATH_ML_INPUT = Path("data", "processed", "ml_input_data.csv")

# roles for data filter
OLINE_PFF_ROLE = "Pass Block"
DLINE_PFF_ROLE = "Pass Rush"
QB_PFF_ROLE = "Pass"
RELEVANT_ROLE = [OLINE_PFF_ROLE, DLINE_PFF_ROLE, QB_PFF_ROLE]

# ml-model
INIT_MODEL_PATH = Path("models", "init_model.pkl")
TESTING_GAME = 2021110100
