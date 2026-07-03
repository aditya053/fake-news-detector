import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def load_data(path):
    """Load and prepare the dataset."""

    df = pd.read_csv(path)

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.fillna("")

    df["content"] = df["title"] + " " + df["text"]

    df["content"] = df["content"].str.replace(
        r"http\S+|www\S+", "", regex=True
    )

    df["content"] = df["content"].str.replace(
        r"[^\w\s]", "", regex=True
    )

    df["content"] = df["content"].str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

    df = df.drop_duplicates(subset=["content"])

    return df


def prepare_features(df):
    """Create TF-IDF features."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X = vectorizer.fit_transform(df["content"])
    y = df["label"]

    return X, y, vectorizer


def train_model(X_train, y_train):
    """Train the SVM classifier."""

    model = LinearSVC(
        C=2.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print()
    print(classification_report(y_test, predictions))


def save_artifacts(model, vectorizer):
    """Save trained model and vectorizer."""

    with open("svm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


def main():

    print("Loading dataset...")

    df = load_data("WELFake_Dataset.csv")

    print(f"Dataset Shape : {df.shape}")

    X, y, vectorizer = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Training model...")

    model = train_model(X_train, y_train)

    print("\nEvaluating model...\n")

    evaluate_model(model, X_test, y_test)

    save_artifacts(model, vectorizer)

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()