import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Step 1: Load and preprocess the data
df = pd.read_csv('keystroke_features.csv')

# Fill missing values (if any) and normalize the data
df.fillna(0, inplace=True)
X = df[['hold_time', 'flight_time']].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the labels (keys) to integers
y = df['key']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

# Step 3: Define a deeper Neural Network architecture
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Step 4: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Save the model
model.save('keystroke_nn_model_deep.h5')
print("Deep Neural Network model saved as 'keystroke_nn_model_deep.h5'.")

# Step 8: Load and Use the Model
from tensorflow.keras.models import load_model

loaded_model = load_model('keystroke_nn_model_deep.h5')
sample_prediction = loaded_model.predict(X_test)
print(f"Sample Prediction: {np.argmax(sample_prediction[:5], axis=1)}")
print(f"Sample Actual: {np.argmax(y_test[:5], axis=1)}")
