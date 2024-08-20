import time
import pandas as pd
from pynput import keyboard
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Data Collection
keystroke_data = []

def on_press(key):
    try:
        keystroke_data.append({
            'key': key.char, 
            'time': time.time(), 
            'event': 'press'
        })
    except AttributeError:
        keystroke_data.append({
            'key': str(key), 
            'time': time.time(), 
            'event': 'press'
        })

def on_release(key):
    keystroke_data.append({
        'key': key.char if hasattr(key, 'char') else str(key), 
        'time': time.time(), 
        'event': 'release'
    })
    if key == keyboard.Key.esc:
        return False

print("Start typing... Press ESC to stop.")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

df = pd.DataFrame(keystroke_data)
df.to_csv('keystroke_data.csv', index=False)
print("Keystroke data saved to 'keystroke_data.csv'.")

# Step 2: Data Preprocessing
df = pd.read_csv('keystroke_data.csv')
hold_times = []
flight_times = []
last_release_time = None

for i in range(len(df)):
    if df.iloc[i]['event'] == 'press':
        release_idx = df[(df['key'] == df.iloc[i]['key']) & (df['event'] == 'release')].index
        if not release_idx.empty:
            hold_time = df.loc[release_idx[0], 'time'] - df.iloc[i]['time']
            hold_times.append(hold_time)
        else:
            hold_times.append(None)
        if last_release_time is not None:
            flight_time = df.iloc[i]['time'] - last_release_time
            flight_times.append(flight_time)
        else:
            flight_times.append(None)
    
    if df.iloc[i]['event'] == 'release':
        last_release_time = df.iloc[i]['time']

df['hold_time'] = pd.Series(hold_times)
df['flight_time'] = pd.Series(flight_times)
df_filtered = df[df['event'] == 'press'].reset_index(drop=True)
df_filtered.to_csv('keystroke_features.csv', index=False)
print("Keystroke features saved to 'keystroke_features.csv'.")

# Step 3: Model Training and Testing
df = pd.read_csv('keystroke_features.csv')
X = df[['hold_time', 'flight_time']].fillna(0)
y = df['key']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 4: Save the trained model to a .pkl file
model_filename = 'keystroke_auth_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(clf, file)
print(f"Trained model saved as '{model_filename}'.")

# Step 5: Load and Use the Model
# To demonstrate, let's reload the model and make a prediction with the test data
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Example: Use the loaded model to make predictions
sample_prediction = loaded_model.predict(X_test)
print(f"Sample Prediction: {sample_prediction[:5]}")
print(f"Sample Actual: {y_test[:5].values}")
