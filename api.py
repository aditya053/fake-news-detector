import pickle 
from fastapi import FastAPI
from utils import preprocess
from pydantic import BaseModel
from scipy.special import expit

class request_format(BaseModel):
    text: str

app=FastAPI()

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.post("/predict")
def predict(request:request_format):
    cleaned_text = preprocess(request.text)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]

    score = model.decision_function(vector)[0]

    probability = expit(score)

    confidence = probability if prediction == 1 else 1 - probability

    return {
        "prediction": "REAL" if prediction == 1 else "FAKE",
        "confidence": round(confidence * 100, 2)
    }