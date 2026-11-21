import numpy as np
import pandas as pd

from src.config import DROP_AFTER_FE


def create_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. log1p for heavy-tailed features
    df["log_measles"] = np.log1p(df["Measles"].clip(lower=0))
    df["log_gdp"] = np.log1p(df["GDP"].clip(lower=0))
    df["log_population"] = np.log1p(df["Population"].clip(lower=0))
    df["log_expenditure"] = np.log1p(df["percentage expenditure"].clip(lower=0))
    df["log_infant_deaths"] = np.log1p(df["infant deaths"].clip(lower=0))
    df["log_under5_deaths"] = np.log1p(df["under-five deaths"].clip(lower=0))
    df["log_hiv"] = np.log1p(df["HIV/AIDS"].clip(lower=0))

    # 2. GDP per capita (key feature)
    df["gdp_per_capita"] = df["GDP"] / df["Population"].replace(0, np.nan)
    df["log_gdp_per_capita"] = np.log1p(df["gdp_per_capita"].clip(lower=0))

    # 3. Immunization - average (3 correlated features, corr > 0.9)
    df["immunization"] = (df["Hepatitis B"] + df["Polio"] + df["Diphtheria"]) / 3

    # 4. Thinness - average (2 nearly identical features, corr ~ 0.99)
    df["thinness_avg"] = (df["thinness 1-19 years"] + df["thinness 5-9 years"]) / 2

    # 5. Child mortality ratio (under5 / infant, shows dynamics)
    df["child_mortality_ratio"] = (
        df["under-five deaths"] / df["infant deaths"].replace(0, np.nan)
    )

    # 6. Epidemic flags
    df["has_measles_outbreak"] = (df["Measles"] > 100).astype(int)
    df["has_hiv_epidemic"] = (df["HIV/AIDS"] > 1).astype(int)

    # 7. Time trend
    df["year_since_2000"] = df["Year"] - 2000

    # 8. Healthcare efficiency
    df["health_expenditure_per_gdp"] = (
        df["Total expenditure"] / df["GDP"].replace(0, np.nan) * 1000
    )

    # 9. Drop source columns (replaced by new ones)
    df = df.drop(columns=DROP_AFTER_FE, errors="ignore")

    return df