from pydantic import BaseModel, Field, ConfigDict


class LifeExpectancy(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Year: int
    Status: str
    Adult_Mortality: float = Field(alias="Adult Mortality")
    infant_deaths: int = Field(alias="infant deaths")
    Alcohol: float
    percentage_expenditure: float = Field(alias="percentage expenditure")
    Hepatitis_B: float = Field(alias="Hepatitis B")
    Measles: int
    BMI: float
    under_five_deaths: int = Field(alias="under-five deaths")
    Polio: float
    Total_expenditure: float = Field(alias="Total expenditure")
    Diphtheria: float
    HIV_AIDS: float = Field(alias="HIV/AIDS")
    GDP: float
    Population: float
    thinness_1_19_years: float = Field(alias="thinness 1-19 years")
    thinness_5_9_years: float = Field(alias="thinness 5-9 years")
    Income_composition_of_resources: float = Field(
        alias="Income composition of resources"
    )
    Schooling: float


class Prediction(BaseModel):
    prediction: float
