import pandas as pd
import os


FEATURE_DIR = os.path.join('..', 'data','features')
RAW_DATA_DIR = os.path.join('..', 'data', 'raw')


def process_data(filepath):
    # Load the collected keystroke data
    # df = pd.read_csv('keystroke_data.csv')
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, filepath))
    

    # Initialize lists to store calculated features
    hold_times = []
    flight_times = []

    # Track the last released key time to calculate flight time
    last_release_time = None

    # Iterate through the DataFrame to calculate features
    for i in range(len(df)):
        if df.iloc[i]['event'] == 'press':
            # Find the corresponding release event for this key
            release_idx = df[(df['key'] == df.iloc[i]['key']) & (df['event'] == 'release')].index
            if not release_idx.empty:
                hold_time = df.loc[release_idx[0], 'time'] - df.iloc[i]['time']
                hold_times.append(hold_time)
            else:
                hold_times.append(None)
            
            # Calculate flight time if there's a previous release time
            if last_release_time is not None:
                flight_time = df.iloc[i]['time'] - last_release_time
                flight_times.append(flight_time)
            else:
                flight_times.append(None)
        
        if df.iloc[i]['event'] == 'release':
            last_release_time = df.iloc[i]['time']

    # Add the calculated features to the DataFrame
    df['hold_time'] = pd.Series(hold_times)
    df['flight_time'] = pd.Series(flight_times)

    # Drop rows where 'event' is 'release', as we only need 'press' events for features
    df_filtered = df[df['event'] == 'press'].reset_index(drop=True)

    # Save the preprocessed data to a new CSV file
    filepath = os.path.join(FEATURE_DIR, file.split('.')[0] + '_features.csv')
    if not os.path.exists(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)
    df_filtered.to_csv(filepath, index=False)
    

if __name__ == "__main__":
    files = os.listdir(RAW_DATA_DIR)
    for file in files:
        process_data(file)
