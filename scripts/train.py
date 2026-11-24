import joblib
import mlflow
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH, MODEL_PATH, EXPERIMENT_NAME, TARGET_COL,
    TEST_SIZE, RANDOM_STATE, N_TRIALS,DROP_COLS
)
from src.logging_config import setup_logging
from src.data.loader import load_path
from src.features.engineering import create_feature_engineering
from src.features.preprocessor import build_preprocessor
from src.models.factory import get_models, get_cv
from src.models.trainer import compare_models, get_best_model
from src.models.tuning import optuna_tune
from src.evaluation.metrics import get_oof_prediction, evaluate_model
from src.evaluation.explainability import create_shap, create_feature_importance

from src.tracking.mlflow_utils import (
    setup_mlflow, log_param, log_metrics, log_artifacts, log_model,
)

logger = setup_logging()

def main():
    setup_mlflow(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Data
        df = load_path(DATA_PATH)
        logger.info(df.head().T)
        logger.info('Describing: %s', df.describe().to_string())
        logger.info('INFO: %s', df.info())

        df = df.drop(columns=DROP_COLS, errors='ignore')
        df = df.dropna(subset=[TARGET_COL])
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,y,test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Feature engineering and preprocessor
        X_train_fe = create_feature_engineering(X_train)
        preprocessor = build_preprocessor(X_train_fe)

        # Comparing models
        models = get_models()
        cv = get_cv()
        results = compare_models(models, preprocessor, cv, X_train, y_train)
        best_name, pipeline = get_best_model(results)

        # Tuning
        pipeline, best_params = optuna_tune(best_name, pipeline, cv, X_train, y_train)

        log_param(best_name, cv, N_TRIALS, best_params)

        # OOF and metrics
        oof_predictions = get_oof_prediction(pipeline, X_train, y_train, cv)
        metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test, oof_predictions)

        log_metrics(metrics)

        # SHAP and feature importance
        shap_figure = create_shap(pipeline, X_train, X_test)
        importance = create_feature_importance(pipeline)

        log_artifacts(importance, shap_figure)

        # Save
        joblib.dump(pipeline, MODEL_PATH)
        log_model(pipeline)

        logger.info('Model saved to %s', MODEL_PATH)

if __name__ == '__main__':
    main()
