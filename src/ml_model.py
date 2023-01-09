import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor

from src.config import PATH_ML_INPUT, TESTING_GAME, INIT_MODEL_PATH


def train_init_model(out: str=INIT_MODEL_PATH):
    """Trains an initial model for expected pocket time (without testing and tuning)."""

    # preprocess (onehot encoding)
    input_data = pd.read_csv(PATH_ML_INPUT)
    input_data = pd.get_dummies(data=input_data, columns=["oline", "dline", "pff_passCoverage"])

    # training
    train_data = input_data[input_data.gameId != TESTING_GAME].drop(labels=["gameId", "playId"], axis=1)
    X_train = train_data.drop(labels="pocket_time", axis=1)
    y_train = train_data.pocket_time
    regression_model = RandomForestRegressor(random_state=0)
    regression_model.fit(X_train, y_train)

    # save model
    with open(out, "wb") as f:
        pickle.dump(regression_model, f)


def evaluate_play(play: int, game: int=TESTING_GAME):
    """Predicts the expected pocket time for a specific play. """
    # model and data
    with open(INIT_MODEL_PATH, "rb") as f:
        regression_model = pickle.load(f)
    input_data = pd.read_csv(PATH_ML_INPUT)
    input_data = pd.get_dummies(data=input_data, columns=["oline", "dline", "pff_passCoverage"])

    # evaluation data
    eval_data = input_data[
        (input_data.gameId == game) & (input_data.playId == play)].drop(labels=["gameId", "playId"], axis=1)
    y_predict = regression_model.predict(eval_data.drop(labels="pocket_time", axis=1))

    # out
    res = pd.read_csv(PATH_ML_INPUT, usecols=["gameId", "playId", "oline", "dline", "pocket_time"])
    res = res[(res.gameId == game) & (res.playId == play)]
    res["expected_pocket_time"] = y_predict
    print(res)



if __name__ == "__main__":
    #train_init_model()
    evaluate_play(play=79)
