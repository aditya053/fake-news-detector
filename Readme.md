# 📰 Fake News Detector

An end-to-end **Fake News Detection** web application that classifies news articles as **Real** or **Fake** using a **Linear Support Vector Machine (LinearSVC)** trained on the **WELFake Dataset**. The application follows a production-style architecture with a **Streamlit frontend**, **FastAPI REST API backend**, and **Docker** containerization.

---

## 📸 Application Preview

> Replace this with your application screenshot.

![Application Screenshot](assets/ui.png)

---

## 🚀 Live Demo

**Frontend:** https://fake-news-detector-hutl.onrender.com

**Backend API:** https://fake-news-api-ise0.onrender.com

**API Documentation:** https://fake-news-api-ise0.onrender.com/docs

---

## ✨ Features

- Classifies news articles as **Real** or **Fake**
- Displays prediction confidence
- Automatic text preprocessing
- REST API powered by FastAPI
- Interactive Streamlit web interface
- Dockerized backend for deployment
- Responsive and user-friendly UI

---

## 🏗️ System Architecture

```text
                 User
                   │
                   ▼
         Streamlit Frontend
                   │
          HTTP POST Request
                   │
                   ▼
        FastAPI REST API
                   │
          Text Preprocessing
                   │
          TF-IDF Vectorization
                   │
          LinearSVC Classifier
                   │
                   ▼
          Prediction Response
```

---

## 🧠 Machine Learning Pipeline

1. User submits a news article through the Streamlit interface.
2. Streamlit sends the article to the FastAPI backend.
3. The backend preprocesses the text by removing URLs, punctuation, and extra whitespace.
4. The cleaned text is transformed using the trained TF-IDF vectorizer.
5. The LinearSVC model predicts whether the article is **Real** or **Fake**.
6. The prediction and confidence score are returned to the frontend.

> **Note:** LinearSVC does not natively produce probabilities. The displayed confidence score is estimated from the model's decision function using a sigmoid transformation.

---

## 🛠️ Tech Stack

### Machine Learning

- Scikit-learn
- LinearSVC
- TF-IDF Vectorizer

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- Streamlit

### Deployment

- Docker
- Render

### Programming Language

- Python

---

## 📂 Project Structure

```text
FakeNewsDetector/
│
├── app.py                    # Streamlit frontend
├── api.py                    # FastAPI backend
├── utils.py                  # Text preprocessing
├── train_model.py            # Model training
├── svm_model.pkl             # Trained classifier
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/fake-news-detector.git

cd fake-news-detector
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

### Start the FastAPI backend

```bash
uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

### Start the Streamlit frontend

```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

Build the Docker image

```bash
docker build -t fake-news-api .
```

Run the container

```bash
docker run -p 8000:8000 fake-news-api
```

---

## 📡 API Endpoint

**POST** `/predict`

### Request

```json
{
  "text": "News article goes here..."
}
```

### Response

```json
{
  "prediction": "Real",
  "confidence": 98.4
}
```

---

## 📊 Model Details

| Item | Value |
|------|------|
| Algorithm | Linear Support Vector Machine (LinearSVC) |
| Feature Extraction | TF-IDF |
| Text Preprocessing | URL removal, punctuation removal, whitespace normalization |
| N-grams | Unigrams + Bigrams |
| Dataset | WELFake Dataset (~72,000 news articles) |
| Test Accuracy | **98.0%** |

---

## 🌐 Deployment

### Frontend

- Streamlit
- Hosted on Render

### Backend

- FastAPI
- Dockerized
- Hosted on Render

---

## 🔮 Future Improvements

- Replace TF-IDF with transformer-based embeddings
- Add multilingual fake news detection
- Improve confidence calibration
- Explain predictions using SHAP
- Add batch prediction support
- Implement CI/CD using GitHub Actions

---

## 📄 License

This project is licensed under the MIT License.