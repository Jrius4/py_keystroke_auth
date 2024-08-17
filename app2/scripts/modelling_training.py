import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

MODELS_DIR = os.path.join('..', 'data','trained_models')
FEATURE_DIR = os.path.join('..', 'data','features')


def train_model(filepath):
    # Load the preprocessed keystroke data
    df = pd.read_csv(os.path.join(FEATURE_DIR, filepath))
   

    # Define the features and target variable
    X = df[['hold_time', 'flight_time']].fillna(0)  # Replace NaN with 0 for simplicity
    y = df['key']  # For demonstration, we're using the key as the target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier()

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    model_filename = 'keystroke_auth_model.pkl'
    filepath = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    with open(filepath,'wb') as file:
        pickle.dump(clf, file)

if __name__ == "__main__":
    files = os.listdir(FEATURE_DIR)
    for file in files:
        train_model(file)
    
