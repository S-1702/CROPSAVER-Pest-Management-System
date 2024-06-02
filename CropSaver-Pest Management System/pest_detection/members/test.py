import cv2
import numpy as np
from tensorflow.keras.applications import SqueezeNet
from tensorflow.keras.applications.squeezenet import preprocess_input, decode_predictions

# Load pre-trained SqueezeNet model
model = SqueezeNet(weights='imagenet')

# Load and preprocess the image
image_path = "insect.png"
img = cv2.imread(image_path)
img = cv2.resize(img, (227, 227))
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Make prediction
predictions = model.predict(img)

# Decode predictions
decoded_predictions = decode_predictions(predictions)

# Print the top prediction
print("Top prediction:", decoded_predictions[0][0])
