import logging

import pandas as pd

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,r2_score,mean_absolute_error
)
from sklearn.model_selection import KFold, cross_val_predict

logger = logging.getLogger(__name__)

def get_oof_prediction(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv: KFold
) -> np.ndarray:
    logger.info('\n\nStart getting oof prediction...')

    oof_predictions = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, n_jobs=-1, method='predict'
    )
    logger.info('Finish getting oof prediction.')
    return oof_predictions


def evaluate_model(
        pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series, oof_predict: np.ndarray
) -> dict:
    logger.info('\n\nStart evaluating model...')

    train_prediction = pipeline.predict(X_train)
    test_prediction = pipeline.predict(X_test)

    metrics = {
        # OOF
        'oof_mae': mean_absolute_error(y_train, oof_predict),
        'oof_rmse': np.sqrt(mean_squared_error(y_train, oof_predict)),
        'oof_r2': r2_score(y_train, oof_predict),

        # Train
        'train_mae': mean_absolute_error(y_train, train_prediction),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_prediction)),
        'train_r2': r2_score(y_train, train_prediction),

        # Test
        'test_mae': mean_absolute_error(y_test, test_prediction),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_prediction)),
        'test_r2': r2_score(y_test, test_prediction)
    }

    logger.info(
        'OOF -> mae=%s, rmse=%s, r2=%s',
        metrics['oof_mae'], metrics['oof_rmse'], metrics['oof_r2']
    )

    logger.info(
        'Train -> mae=%s, rmse=%s, r2=%s',
        metrics['train_mae'], metrics['train_rmse'], metrics['train_r2']
    )

    logger.info(
        'Test -> mae=%s, rmse=%s, r2=%s',
        metrics['test_mae'], metrics['test_rmse'], metrics['test_r2']
    )
    logger.info('Finish evaluating model.')
    return metrics