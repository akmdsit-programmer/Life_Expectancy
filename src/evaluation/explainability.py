import logging

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)

def create_shap(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    logger.info('\n\nStart creating SHAP plot...')

    model = pipeline.named_steps['model']

    X_train_fe = pipeline.named_steps['features'].transform(X_train)
    X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train_fe)

    X_test_fe = pipeline.named_steps['features'].transform(X_test)
    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test_fe)

    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    X_processed = pd.DataFrame(X_test_processed, columns=feature_names)

    if isinstance(model, (RandomForestRegressor, LGBMRegressor, XGBRegressor, CatBoostRegressor)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, Ridge):
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        background = shap.sample(X_train_df, 100)
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.Explainer(model)

    shap_values = explainer(X_processed)

    # Summary (beeswarm) plot
    fig_summary, ax_summary = plt.subplots()
    shap.plots.beeswarm(shap_values, ax=ax_summary, show=False, plot_size=None)
    fig_summary.suptitle('Summary Plot')
    fig_summary.tight_layout()

    # Bar plot
    fig_bar, ax_bar = plt.subplots()
    shap.plots.bar(shap_values, ax=ax_bar, show=False)
    fig_bar.suptitle('Bar Plot')
    fig_bar.tight_layout()

    # Waterfall plot
    shap.plots.waterfall(shap_values[0], show=False)
    fig_waterfall = plt.gcf()
    fig_waterfall.axes[0].set_title('Waterfall Plot')
    fig_waterfall.tight_layout()

    logger.info('Finish creating SHAP plot.')
    return {
        'Summary plot': fig_summary,
        'Bar Plot': fig_bar,
        'Waterfall Plot': fig_waterfall
    }

def create_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    logger.info('\n\nStart creating feature importance...')

    model = pipeline.named_steps['model']

    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    if hasattr(model, 'feature_importances_'):
        values = model.feature_importances_
    elif hasattr(model, 'coef_'):
        values = np.abs(model.coef_).ravel()
    else:
        raise AttributeError(f"{type(model).__name__} has no feature_importances_ / coef_")

    df = pd.DataFrame({
        'Features': feature_names,
        'Importance': values
    })

    df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
    logger.info("Top 10 features:\n%s", df.head(10).to_string())
    logger.info('Finish creating feature importance.')
    return df
