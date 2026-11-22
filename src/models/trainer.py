import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold

from src.models.factory import build_pipeline

logger = logging.getLogger(__name__)

def compare_models(
    models: dict[str, BaseEstimator], preprocessor, cv: KFold,
    X_train: pd.DataFrame, y_train: pd.Series,
) -> dict:
    logger.info('Start comparing models...')

    results = {}

    for name, model in models.items():
        logger.info(f'Comparing {name}...')

        pipeline = build_pipeline(model, preprocessor)

        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, n_jobs=-1,
            scoring='neg_root_mean_squared_error', error_score='raise'
        )
        rmse_score = -scores

        results[name] = {
            'mean_rmse': np.mean(rmse_score),
            'std_rmse': np.std(rmse_score),
            'pipeline': pipeline
        }
        logger.info('Finished %s: RMSE=%s', name, np.mean(rmse_score))
    logger.info('Finished comparing models.')
    return results

def get_best_model(results: dict) -> tuple[str, object]:
    logger.info('Start getting best model...')
    best_name = min(results, key=lambda x: results[x]['mean_rmse'])
    pipeline = results[best_name]['pipeline']
    logger.info('Best model: %s (RMSE=)' , best_name, results[best_name]['mean_rmse'])
    logger.info('Finished getting best model.')
    return best_name, pipeline
