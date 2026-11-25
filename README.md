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


## API Example

**Request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Year": 2015,
    "Status": "Developed",
    "Adult Mortality": 75,
    "infant deaths": 3,
    "Alcohol": 9.5,
    "percentage expenditure": 4500.0,
    "Hepatitis B": 95,
    "Measles": 10,
    "BMI": 25.5,
    "under-five deaths": 4,
    "Polio": 96,
    "Total expenditure": 8.5,
    "Diphtheria": 96,
    "HIV/AIDS": 0.1,
    "GDP": 35000.0,
    "Population": 5000000,
    "thinness 1-19 years": 1.2,
    "thinness 5-9 years": 1.0,
    "Income composition of resources": 0.9,
    "Schooling": 16.0
  }'
```

**Response:**

```json
{
  "prediction": 84.29647813321405
}
```