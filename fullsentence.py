import numpy as np
import cv2
import os
import tensorflow as tf
load_model = tf.keras.models.load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt

# 🔹 Load Trained Model
MODEL_PATH = "handwriting_letter_recognition.h5"
model = load_model(MODEL_PATH)

# 🔹 Define Label Encoder (A-Z)
letters = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]  # A-Z and a-z
encoder = LabelEncoder()
encoder.fit(letters)

# 🔹 Character Segmentation Using OpenCV
def segment_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 10:  # Ignore noise
            char = img[y:y+h, x:x+w]
            char = cv2.resize(char, (28, 28))  # Resize for CNN input
            char_images.append(char)
            bounding_boxes.append((x, y, w, h))

    # Sort characters from left to right
    sorted_chars = [char for _, char in sorted(zip(bounding_boxes, char_images), key=lambda b: b[0])]
    return sorted_chars


# 🔹 Predict Letters Using the CNN Model
def recognize_sentence(image_path):
    characters = segment_characters(image_path)

    sentence = ""
    for char in characters:
        char = np.array(char).reshape(1, 28, 28, 1).astype("float32") / 255.0

        prediction = model.predict(char)
        predicted_label = encoder.inverse_transform([np.argmax(prediction)])
        sentence += predicted_label[0]

    print(f"Recognized Sentence: {sentence}")
    return sentence

# 🔹 Test with a Sample Image
image_path = "./iam_dataset/data_subset/img-19.png"
recognized_text = recognize_sentence(image_path)

# 🔹 Display the Image
img = Image.open(image_path)
plt.imshow(img, cmap="gray")
plt.title(f"Recognized Text: {recognized_text}")
plt.show()
