import time
import keyboard
import pandas as pd
import os
from datetime import datetime

DATA_DIR = os.path.join('..', 'data', 'raw')

def record_keystrokes():
    print("Start typing. Press 'Esc' to stop.")
    records = []

    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            record = {
                'key': event.name,
                'timestamp': time.time()
            }
            records.append(record)

        if event.name == 'esc':
            break

    return pd.DataFrame(records)

def save_data(data, filename='keystrokes'):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(DATA_DIR, f'{filename}_{timestamp}.csv')
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    keystrokes_df = record_keystrokes()
    save_data(keystrokes_df)
