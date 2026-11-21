from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "life_expectancy_data.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_TRIALS = 20

EXPERIMENT_NAME = "life_expectancy_data"
REGISTERED_MODEL_NAME = "life_expectancy_data"

TARGET_COL = "Life expectancy"

DROP_COLS = ["Country"]

DROP_AFTER_FE = []