import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

PROCESSED_DATA_DIR = os.path.join('..', 'data', 'processed')
MODEL_DIR = os.path.join('..', 'models')

PROCESSED_DATA_DIR_MODERN = Path('../data/processed')
MODEL_DIR_MODERN = Path('../models')

# def load_features() -> tuple[np.ndarray, np.ndarray]:
#     features = []
#     labels = []
    
#     for file in PROCESSED_DATA_DIR.glob('*.csv'):
#         feature_data = pd.read_csv(file, index_col=0)
#         features.append(feature_data.values)

#         # Ensure there's a label for each feature set
#         labels.extend([1] * feature_data.shape[0])  # Assuming '1' as a placeholder

#     return np.array(features), np.array(labels)

def load_features():
    features = np.array([])
    labels = []
    files = os.listdir(PROCESSED_DATA_DIR)
    for file in files:  # Iterate over all processed feature files
        feature_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, file), index_col=0)
        features = np.append(features,feature_data.values) 
        labels.extend([1] * feature_data.shape[0]) # Replace with actual labels in a real use case
    T = features
    J = np.array(labels)
    X = T.reshape(-1,1)
    return X,J

def train_model(X, y):
    print(['xShape',X.shape,'yShape',y.shape])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

def save_model(model, filename='keystroke_model.pkl'):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    X, y = load_features()
    model = train_model(X, y)
    save_model(model)
