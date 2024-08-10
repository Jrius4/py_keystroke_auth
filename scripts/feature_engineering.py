import pandas as pd
import numpy as np
import os

RAW_DATA_DIR = os.path.join('..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join('..', 'data', 'processed')

def load_data(filepath):
    return pd.read_csv(filepath)

def extract_features(data):
    data['time_diff'] = data['timestamp'].diff().fillna(0)
    
    key_hold_times = data.groupby('key')['time_diff'].mean()
    key_intervals = data['time_diff'].mean()
    
    features = key_hold_times._append(pd.Series(key_intervals, index=['mean_interval']))
    # features = pd.concat([key_hold_times,pd.Series(key_intervals, index=['mean_interval'])])
    return features

def save_features(features, filename='features'):
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    filepath = os.path.join(PROCESSED_DATA_DIR, f'{filename}.csv')
    features.to_csv(filepath, index=True)
    print(f"Features saved to {filepath}")

if __name__ == "__main__":
    files = os.listdir(RAW_DATA_DIR)
    for file in files:
        data = load_data(os.path.join(RAW_DATA_DIR, file))
        features = extract_features(data)
        save_features(features, filename=file.split('.')[0])
