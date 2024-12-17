# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report

# Set dataset directory
DATASET_DIR = 'C:\\Users\\Blade\\OneDrive\\Documentos\\Github\\CECS456_dataset\\medical_mnist'  # Path to unzipped Medical MNIST folder (change according to path in own device)

# Classes based on Medical MNIST dataset
CLASSES = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 64  # All images are 64x64 in this dataset

# 1. Load and Preprocess the Data
def load_dataset(dataset_dir):
    images = []
    labels = []
    for idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img_resized)
            labels.append(idx)  # Use index as label
    
    # Convert to NumPy arrays
    images = np.array(images) / 255.0  # Normalize pixel values to [0, 1]
    labels = np.array(labels)
    
    # Add channel dimension for CNN
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return images, labels

# Load the dataset
print("Loading dataset...")
images, labels = load_dataset(DATASET_DIR)
print(f"Loaded {len(images)} images.")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 2. Define the CNN Model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

# Create the model
model = create_cnn_model((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train the Model
print("Training model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 4. Evaluate the Model
print("Evaluating model on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 5. Test the Model on Example Cases
def predict_example_images(model, X_test, y_test, class_names):
    # Select 5 random test images
    indices = np.random.choice(len(X_test), 5, replace=False)
    for i in indices:
        img = X_test[i]
        true_label = class_names[y_test[i]]
        
        # Predict
        pred = model.predict(np.expand_dims(img, axis=0))
        pred_label = class_names[np.argmax(pred)]
        
        # Plot image
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
        plt.show()

# Run predictions on test examples
predict_example_images(model, X_test, y_test, CLASSES)

# Additional Test Cases
# Test Case 1: Random Test Images
def test_random_images(model, X_test, y_test, class_names, num_images=5):
    indices = np.random.choice(len(X_test), num_images, replace=False)
    for i in indices:
        img = X_test[i]
        true_label = class_names[y_test[i]]
        
        # Predict
        pred = model.predict(np.expand_dims(img, axis=0))
        pred_label = class_names[np.argmax(pred)]
        
        # Plot image
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
        plt.show()

# Run the test
test_random_images(model, X_test, y_test, CLASSES)

# Test Case 2: Performance Metrics
def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES))

# Run the evaluation
evaluate_model_performance(model, X_test, y_test)
