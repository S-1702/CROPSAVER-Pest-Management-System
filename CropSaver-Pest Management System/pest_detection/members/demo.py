import tensorflow as tf
import numpy as np
import os
import glob

# Define the path to the dataset directory
data_dir = 'path/to/dataset'

# Define the batch size and number of classes
batch_size = 32
num_classes = len(glob.glob(os.path.join(data_dir, '*')))

# Define the image size and number of channels
image_size = (224, 224)
num_channels = 3

# Define the data augmentation pipeline
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Define the image preprocessing function
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=num_channels)
  image = tf.image.resize(image, image_size)
  image = tf.cast(image, tf.float32) / 255.0
  return image

# Define the dataset loading function
def load_dataset(data_dir, batch_size, image_size, num_channels, num_classes, data_augmentation):
  # Define the file pattern for the dataset
  file_pattern = os.path.join(data_dir, '*/*.jpg')

  # Define the dataset using the file pattern
  dataset = tf.data.Dataset.list_files(file_pattern)

  # Define the map function to preprocess each image
  def map_fn(file_path):
    image = tf.io.read_file(file_path)
    image = preprocess_image(image)
    label = tf.strings.split(file_path, os.path.sep)[-2]
    label = tf.cast(tf.equal(tf.range(num_classes), tf.cast(label, tf.int64)), tf.int64)
    return image, label

  # Apply the map function to the dataset
  dataset = dataset.map(map_fn)

  # Shuffle the dataset
  dataset = dataset.shuffle(buffer_size=10000)

  # Apply the data augmentation pipeline
  dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

  # Batch the dataset
  dataset = dataset.batch(batch_size)

  # Prefetch the dataset
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  return dataset

# Load the dataset
train_dataset = load_dataset(data_dir, batch_size, image_size, num_channels, num_classes, data_augmentation)

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], num_channels)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model with a suitable optimizer, loss function, and evaluation metric
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model on the training set, using the validation set to tune hyperparameters and prevent overfitting
model.fit(train_dataset,
          epochs=10,
          validation_data=val_dataset)

# Evaluate the model on the testing set and calculate metrics such as accuracy, precision, recall, and F1 score
test_loss, test_acc = model.evaluate(test_dataset)
test_pred = model.predict(test_dataset)
test_pred_labels = tf.argmax(test_pred, axis=1)
test_true_labels = tf.argmax(test_dataset.map(lambda x, y: y), axis=1)
precision = tf.reduce_mean(tf.cast(tf.equal(test_pred_labels, test_true_labels), tf.float32))
recall = tf.reduce_mean(tf.cast(tf.equal(test_pred_labels, test_true_labels), tf.float32))
f1_score = 2 * precision * recall / (precision + recall)