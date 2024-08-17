import time
import pandas as pd
import keyboard
# from pynput import keyboard
from datetime import datetime
import os


DATA_DIR = os.path.join('..', 'data', 'raw')




def record_keystrokes():
    print("Start typing. Press 'Esc' to stop.")
    record = []
    # Create a listener
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            record.append({
                'key': event.name, 
                'time': time.time(), 
                'event': 'press'
            })
        if event.event_type == keyboard.KEY_UP:
            record.append({
                'key': event.name, 
                'time': time.time(), 
                'event': 'release'
            })
        if event.name == 'esc':
            break
            # if event.name == 'esc':
            #     break

    # Convert the collected keystroke data to a DataFrame
    df = pd.DataFrame(record)
    return df
    
    

def save_data(data,filename="keystrokes"):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(DATA_DIR, f"{filename}_{timestamp}.csv")
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


if __name__ == '__main__':
    keyboard_dataFrame = record_keystrokes()
    save_data(keyboard_dataFrame)