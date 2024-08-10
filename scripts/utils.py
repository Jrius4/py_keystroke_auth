import pandas as pd
import os

def load_csv_files(directory):
    files = os.listdir(directory)
    data_frames = []
    for file in files:
        if file.endswith('.csv'):
            data_frames.append(pd.read_csv(os.path.join(directory, file)))
    return pd.concat(data_frames, ignore_index=True)
