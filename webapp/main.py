from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd

app = Flask(__name__)
DB_NAME = 'keystroke_auth.db'

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            model_pkl BLOB,
            model_h5 TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to extract features (flight time and delay time)
def extract_features(keystrokes):
    press_times = []
    release_times = []
    for k in keystrokes:
        if k['type'] == 'keydown':
            press_times.append(k['time'])
        elif k['type'] == 'keyup':
            release_times.append(k['time'])
    
    # Flight time (time between press and release of the same key)
    flight_times = np.array(release_times) - np.array(press_times)
    
    # Delay time (time between release of one key and press of the next key)
    delay_times = np.array(press_times[1:]) - np.array(release_times[:-1])
    
    return np.concatenate([flight_times, delay_times])


def store_collected_data(keystore_data,username):
    df = pd.DataFrame(keystore_data)
    model_dir = os.path.join('data','raw')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filepath = os.path.join(model_dir, f'{username}.csv')
    df.to_csv(filepath,index=False)

# Function to save a model and return the file path
def save_model(model, username):
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'{username}.h5')
    model.save(model_path)
    return model_path

# Create a TensorFlow model
def create_tf_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    form_data = request.json
    username = form_data['username']
    keystrokes = form_data['keystrokes']
  

    features = extract_features(keystrokes)
    features = features.reshape(1, -1)
    

    
    

    # Train a simple RandomForest model as a backup
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    rf_model = RandomForestClassifier()
    rf_model.fit(features, np.array([1]))  # Dummy target value

    # Save the RandomForest model as a pickle
    model_pkl = pickle.dumps(rf_model)

    # Train and save a TensorFlow model
    tf_model = create_tf_model(features.shape[1])
    tf_model.fit(features, np.array([1]), epochs=10, verbose=0)  # Dummy target value
    model_h5_path = save_model(tf_model, username)

    # Save user data into SQLite
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, model_pkl, model_h5) VALUES (?, ?, ?)',
                  (username, model_pkl, model_h5_path))
        conn.commit()
        store_collected_data(keystrokes,username)
    except sqlite3.IntegrityError:
        return "Username already exists!", 400
    finally:
        conn.close()

    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    form_data = request.json
    # username = form_data['username']
    # keystrokes = form_data['keystrokes']
    username = form_data['username']
    keystrokes = form_data['keystrokes']

    features = extract_features(keystrokes)
    features = features.reshape(1, -1)

    # Retrieve user data
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT model_pkl, model_h5 FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()

    if result:
        # Load the RandomForest model from pickle
        model_pkl = result[0]
        rf_model = pickle.loads(model_pkl)

        # Load the TensorFlow model
        model_h5_path = result[1]
        tf_model = tf.keras.models.load_model(model_h5_path)

        # Make predictions
        prediction_rf = rf_model.predict(features)
        prediction_tf = tf_model.predict(features)
        
        print(f'RandomForest Prediction: {prediction_rf}')
        print(f'TensorFlow Prediction: {prediction_tf}')

        # Simple voting mechanism
        if prediction_rf == 1 and prediction_tf > 0.5:
            return 'Login successful!'
        else:
            return 'Authentication failed!', 403
    else:
        return 'User not found!', 404

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
