from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the model
model = load_model('model.h5')
print("Model Loaded Successfully")

def classify(img_file):
    img_name = img_file
    try:
        # Load and preprocess the image
        test_image = image.load_img(img_name, target_size=(256, 256), color_mode='grayscale')
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict the class
        result = model.predict(test_image)
        arr = np.array(result[0])
        print(arr)
        max_prob = arr.argmax(axis=0)
        classes = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
        predicted_class = classes[max_prob]
        print(img_name, predicted_class)
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

# Define the dataset path
path = os.path.join('D:', 'MasterClass', 'Artificial_Intelligence', 'Day13', 'Dataset', 'val', 'TWO')
files = []

# Collect all image files
for r, d, f in os.walk(path):
    for file in f:
        if file.endswith('.png'):
            files.append(os.path.join(r, file))

# Classify each image
for f in files:
    classify(f)
    print('\n')