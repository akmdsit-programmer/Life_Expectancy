import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


from src.features.engineering import create_feature_engineering

feature_engineering = FunctionTransformer(create_feature_engineering, validate=False)
feature_engineering.set_output(transform='pandas')

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('numerical', numeric_transformer, numeric_columns),
        ('categorical', categorical_transformer, categorical_columns)
    ])