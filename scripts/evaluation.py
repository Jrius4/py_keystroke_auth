import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report

MODEL_DIR = os.path.join('..', 'models')
PROCESSED_DATA_DIR = os.path.join('..', 'data', 'processed')

def load_model(filepath):
    return joblib.load(filepath)

def load_features():
    features = []
    labels = []
    
    files = os.listdir(PROCESSED_DATA_DIR)
    for file in files:
        feature_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, file), index_col=0)
        features.append(feature_data.values)
        labels.append(1)  # Replace with actual labels
    
    return np.array(features), np.array(labels)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred)
    print(report)

if __name__ == "__main__":
    model = load_model(os.path.join(MODEL_DIR, 'keystroke_model.pkl'))
    X, y = load_features()
    evaluate_model(model, X, y)
