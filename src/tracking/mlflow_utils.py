import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import REGISTERED_MODEL_NAME

def setup_mlflow(experiment_name: str) -> None:
    mlflow.set_experiment(experiment_name)

def log_param(best_name: str, cv, n_trials: int, best_params: dict) -> None:
    mlflow.log_param('best_model', best_name)
    mlflow.log_param('cv_folds', cv.get_n_splits())
    mlflow.log_param('optuna_trials', n_trials)
    mlflow.log_params(best_params)

def log_metrics(metrics: dict) -> None:
    for key in [
        'oof_mae', 'oof_rmse', 'oof_r2',
        'train_mae', 'train_rmse', 'train_r2',
        'test_mae', 'test_rmse', 'test_r2'
    ]:
        mlflow.log_metric(key, metrics[key])

def log_artifacts(importance: pd.DataFrame, shap_figure: dict) -> None:
    mlflow.log_text(importance.to_csv(index=False), 'feature_importance.csv')
    mlflow.log_figure(shap_figure['Summary plot'], 'shap_summary.png')
    mlflow.log_figure(shap_figure['Bar Plot'], 'shap_bar.png')
    mlflow.log_figure(shap_figure['Waterfall Plot'], 'shap_waterfall.png')

def log_model(pipeline: Pipeline) -> None:
    model_info = mlflow.sklearn.log_model(
        pipeline,
        name='model',
        serialization_format='cloudpickle',
        pip_requirements=None,
        registered_model_name=REGISTERED_MODEL_NAME
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias='production',
        version=model_info.registered_model_version
    )