import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import string

to_categorical = tf.keras.utils.to_categorical
Adam = tf.keras.optimizers.Adam
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Sequential = tf.keras.models.Sequential

# 📥 Load EMNIST Dataset (Letters only)
dataset, info = tfds.load("emnist/balanced", split="train", as_supervised=True, with_info=True)

# Convert dataset to NumPy arrays
X_data, y_data = [], []

for image, label in dataset:
    img = tf.image.resize(image, (28, 28))  # Resize to 28x28
    img = tf.cast(img, tf.float32) / 255.0  # Normalize
    X_data.append(img.numpy())
    
    # Here, assign labels for both upper and lower cases (e.g., 0-25 for 'A-Z', 26-51 for 'a-z')
    label_index = label.numpy() - 1  # Adjust label index to 0-25 for lowercase
    y_data.append(label_index)  # Assuming the dataset is lowercase only

letters = list(string.ascii_uppercase + string.ascii_lowercase)  # ['A', 'B', ... 'Z', 'a', 'b', ... 'z']

num_classes = len(letters) 
X_data = np.array(X_data).reshape(-1, 28, 28, 1)
y_data = to_categorical(np.array(y_data), num_classes=num_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(f"Train images: {X_train.shape}, Train labels: {y_train.shape}")
print(f"Test images: {X_test.shape}, Test labels: {y_test.shape}")

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")  # Updated for 52 classes
])


model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training vs Validation Loss")

plt.show()

model.save("handwriting_letter_recognition.h5")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
