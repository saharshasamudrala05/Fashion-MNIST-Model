# Fashion-MNIST-Model
Overview
This repository contains a deep learning model built to classify images from the Fashion MNIST dataset. The model achieves an accuracy of 91.31% on the test set.

Table of Contents
Dataset
Requirements
Model Architecture
Training the Model
Results
Usage
License
Dataset
The Fashion MNIST dataset consists of 60,000 training images and 10,000 testing images of 10 different fashion categories, such as T-shirts, trousers, dresses, and more.

Kaggle Fashion MNIST Dataset
Requirements
To run the model, make sure you have the following Python packages installed:

bash
Copy code
pip install numpy pandas tensorflow matplotlib
Model Architecture
The model is built using TensorFlow and Keras. Below is a summary of the architecture:

Input Layer: 28x28 pixel images (grayscale).
Convolutional Layer: 32 filters of size 3x3, followed by ReLU activation.
Max Pooling Layer: 2x2 pool size.
Dropout Layer: 25% dropout rate.
Flatten Layer: Flatten the output from the previous layer.
Dense Layer: 128 units with ReLU activation.
Output Layer: 10 units with softmax activation for classification.
Training the Model
Here’s the code snippet for training the model:

python
Copy code
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
Results
After training the model for 10 epochs, it achieved an accuracy of 91.31% on the test dataset.

Usage
To run the model:

Clone the repository.
Ensure all required libraries are installed.
Run the training script.
License
This project is licensed under the MIT License.

