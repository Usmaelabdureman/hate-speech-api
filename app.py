from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_pipeline 
from fastapi.middleware.cors import CORSMiddleware# Importing the function from your model module

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL during deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str  # Adjust this based on your model's prediction output

@app.get("/")
def home():
    return "Welcome to the API!"

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    predicted_label = predict_pipeline(payload.text)
    return {"label": predicted_label}
