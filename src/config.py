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

DROP_AFTER_FE = [
    "Measles",                    # → log_measles + has_measles_outbreak
    "GDP",                        # → log_gdp + gdp_per_capita
    "Population",                 # → log_population + gdp_per_capita
    "percentage expenditure",     # → log_expenditure
    "infant deaths",              # → log_infant_deaths + child_mortality_ratio
    "under-five deaths",          # → log_under5_deaths + child_mortality_ratio
    "HIV/AIDS",                   # → log_hiv + has_hiv_epidemic
    "Hepatitis B",                # → immunization
    "Polio",                      # → immunization
    "Diphtheria",                 # → immunization
    "thinness 1-19 years",        # → thinness_avg
    "thinness 5-9 years",         # → thinness_avg
    "Year",                       # → year_since_2000
]