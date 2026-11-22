from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.config import RANDOM_STATE, N_SPLITS
from src.features.preprocessor import feature_engineering


def get_models() -> dict[str, BaseEstimator]:
    return {
        'Ridge': Ridge(),
        "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_STATE),
        "XGBRegressor": XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        'LGBMRegressor': LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1),
        "CatBoostRegressor": CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
    }


def build_pipeline(model: BaseEstimator, preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("features", feature_engineering),
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def get_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)