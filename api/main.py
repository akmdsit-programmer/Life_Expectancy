import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.schemas import LifeExpectancy, Prediction
from src.config import MODEL_PATH

model = None

@asynccontextmanager
async def lifespan(api: FastAPI):
    global model

    model = joblib.load(MODEL_PATH)

    yield

app = FastAPI(title='Life Expectancy API', lifespan=lifespan)

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': model is not None
    }

@app.post('/predict', response_model=Prediction)
def predict(data: LifeExpectancy):
    df = pd.DataFrame([data.model_dump(by_alias=True)])

    prediction = float(model.predict(df)[0])

    return {
        'prediction': prediction
    }