# Life Expectancy Prediction

End-to-end Machine Learning project for predicting life expectancy based on demographic, economic, and health-related indicators across different countries and years.

## Tech Stack

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, CatBoost, XGBoost, LightGBM
- **Hyperparameter Optimization:** Optuna
- **Explainability:** SHAP
- **Experiment Tracking:** MLflow
- **API:** FastAPI
- **Testing:** Pytest
- **Deployment:** Docker
- **Version Control:** Git, GitHub

## Dataset

The dataset contains demographic, economic, and health-related indicators for different countries across multiple years.

### Features

- `Country` — country name
- `Year` — year of observation
- `Status` — development status (Developing / Developed)
- `Adult Mortality` — adult mortality rate
- `infant deaths` — number of infant deaths
- `Alcohol` — alcohol consumption
- `percentage expenditure` — health expenditure as a percentage of GDP
- `Hepatitis B` — Hepatitis B immunization coverage
- `Measles` — number of reported measles cases
- `BMI` — average Body Mass Index
- `under-five deaths` — number of deaths under age five
- `Polio` — Polio immunization coverage
- `Total expenditure` — government health expenditure
- `Diphtheria` — Diphtheria immunization coverage
- `HIV/AIDS` — HIV/AIDS-related mortality
- `GDP` — Gross Domestic Product per capita
- `Population` — population size
- `thinness 1-19 years` — prevalence of thinness among people aged 10–19
- `thinness 5-9 years` — prevalence of thinness among children aged 5–9
- `Income composition of resources` — Human Development Index-related income composition indicator
- `Schooling` — average years of schooling

### Target Variable

- `Life expectancy` — expected average lifespan in years

## Project Structure

- `src/` — ML pipeline
- `data/` — dataset
- `models/` — trained models
- `scripts/` — training scripts
- `api/` — FastAPI service
- `tests/` — automated tests

## Installation

```bash
pip install -r requirements.txt
```

## Project Status

In development.