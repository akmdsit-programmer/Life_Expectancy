import logging

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, KFold
import optuna

logger = logging.getLogger(__name__)


def _suggest_ridge(trial) -> dict:
    return {
        "model__alpha": trial.suggest_float("model__alpha", 1e-4, 1000, log=True),
        "model__fit_intercept": trial.suggest_categorical("model__fit_intercept", [True, False]),
        "model__solver": trial.suggest_categorical(
            "model__solver", ["auto", "svd", "cholesky", "lsqr"]
        ),
    }


def _suggest_rf(trial) -> dict:
    return {
        "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 500, step=50),
        "model__max_depth": trial.suggest_int("model__max_depth", 5, 30),
        "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
        "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 10),
        "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", "log2", 1.0]),
    }


def _suggest_xgb(trial) -> dict:
    return {
        "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 1000, step=50),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 12),
        "model__min_child_weight": trial.suggest_int("model__min_child_weight", 1, 10),
        "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
        "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.6, 1.0),
        "model__gamma": trial.suggest_float("model__gamma", 0, 5),
        "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10, log=True),
        "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10, log=True),
    }

def _suggest_lgbm(trial) -> dict:
    return {
        "model__n_estimators": trial.suggest_int("model__n_estimators", 100, 1000, step=50),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
        "model__num_leaves": trial.suggest_int("model__num_leaves", 15, 255),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 12),
        "model__min_child_samples": trial.suggest_int("model__min_child_samples", 5, 100),
        "model__subsample": trial.suggest_float("model__subsample", 0.6, 1.0),
        "model__subsample_freq": trial.suggest_int("model__subsample_freq", 1, 10),
        "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.6, 1.0),
        "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10, log=True),
        "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10, log=True),
        "model__min_split_gain": trial.suggest_float("model__min_split_gain", 0, 1),
    }

def _suggest_catboost(trial) -> dict:
    return {
        "model__iterations": trial.suggest_int("model__iterations", 200, 1000, step=50),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
        "model__depth": trial.suggest_int("model__depth", 4, 10),
        "model__l2_leaf_reg": trial.suggest_float("model__l2_leaf_reg", 1, 20, log=True),
        "model__random_strength": trial.suggest_float("model__random_strength", 0, 10),
        "model__bagging_temperature": trial.suggest_float("model__bagging_temperature", 0, 10),
        "model__border_count": trial.suggest_int("model__border_count", 32, 255),
    }

SUGGESTERS = {
    "Ridge": _suggest_ridge,
    "RandomForestRegressor": _suggest_rf,
    "XGBRegressor": _suggest_xgb,
    "LGBMRegressor": _suggest_lgbm,
    "CatBoostRegressor": _suggest_catboost
}

def optuna_tune(
    best_name: str, pipeline, cv: KFold,
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20
):
    logger.info('\n\nStart optuna tuning...')

    if best_name not in SUGGESTERS:
        raise ValueError(f'Unknown model: {best_name}')

    def objective(trial):
        params = SUGGESTERS[best_name](trial)
        trial_pipeline = clone(pipeline)
        trial_pipeline.set_params(**params)
        scores = cross_val_score(
            trial_pipeline, X_train, y_train, cv=cv, n_jobs=-1,
            scoring='neg_root_mean_squared_error', error_score='raise'
        )
        return -scores.mean()


    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    logger.info('Best params: %s', study.best_params)
    logger.info('RMSE: %.4f', study.best_value)

    pipeline.set_params(**study.best_params)
    pipeline.fit(X_train, y_train)
    return pipeline, study.best_params

