import joblib
import keyboard
import time
import numpy as np

# Load the trained model
model = joblib.load('keystroke_auth_model.pkl')

def capture_keystroke_data():
    keystrokes = []
    print("Start typing...")

    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            keystrokes.append((event.name, time.time()))

        if event.name == 'esc':
            break

    return keystrokes

def preprocess_keystroke_data(keystrokes):
    hold_times = []
    flight_times = []

    for i in range(len(keystrokes) - 1):
        if keystrokes[i][0] != 'esc':
            hold_time = keystrokes[i + 1][1] - keystrokes[i][1]
            hold_times.append(hold_time)

            if i > 0:
                flight_time = keystrokes[i][1] - keystrokes[i - 1][1]
                flight_times.append(flight_time)

    return hold_times, flight_times

# Capture keystroke data
keystrokes = capture_keystroke_data()

# Preprocess the data
hold_times, flight_times = preprocess_keystroke_data(keystrokes)
X_new = [hold_times + flight_times]

# Ensure the data format is correct for the model
X_new = np.array(X_new).reshape(1, -1)

# Make a prediction
prediction = model.predict(X_new)

# Authenticate the user
user_class = 1  # Define the correct class for an authorized user
if prediction == user_class:
    print("Authentication successful!")
else:
    print("Authentication failed!")
