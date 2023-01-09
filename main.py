
from src.data_preprocessing import get_pocket_stability_data, prepare_ml_data
from src.ml_model import train_init_model


PREPARE_DATA = False
TRAIN_MODEL = True

if __name__ == "__main__":
    if PREPARE_DATA:
        get_pocket_stability_data()
    if TRAIN_MODEL:
        prepare_ml_data()
        train_init_model()
