import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Dataset paths
DATASET_DIR = "C:\CECS456FinalProject"
CATEGORIES = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
IMG_SIZE = 64  # Resize all images to 64x64

# Step 1: Load and Preprocess the Data
def load_data(dataset_dir, categories):
    data = []
    labels = []

    for idx, category in enumerate(categories):
        category_path = os.path.join(dataset_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                # Load and resize image
                img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = tf.keras.utils.img_to_array(img) / 255.0
                data.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(data), np.array(labels)

# Load the dataset
data, labels = load_data(DATASET_DIR, CATEGORIES)

# Split into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Step 2: Create Bags for MIP
def create_bags(data, labels, bag_size=10):
    bags = []
    bag_labels = []
    num_samples = len(labels)

    for _ in range(num_samples // bag_size):
        indices = np.random.choice(range(num_samples), bag_size, replace=False)
        bag = data[indices]
        bag_label = labels[indices[0]]  # Assign the label of the first instance in the bag
        bags.append(bag)
        bag_labels.append(bag_label)

    return np.array(bags), np.array(bag_labels)

train_bags, train_bag_labels = create_bags(train_data, train_labels, bag_size=10)
test_bags, test_bag_labels = create_bags(test_data, test_labels, bag_size=10)

# Step 3: Build the MIP Model
def build_mip_model(input_shape):
    instance_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])

    bag_input = layers.Input(shape=(None, *input_shape))
    instance_embeddings = layers.TimeDistributed(instance_model)(bag_input)
    bag_representation = layers.GlobalMaxPooling1D()(instance_embeddings)
    output = layers.Dense(len(CATEGORIES), activation='softmax')(bag_representation)

    model = models.Model(inputs=bag_input, outputs=output)
    return model

mip_model = build_mip_model((IMG_SIZE, IMG_SIZE, 3))
mip_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = mip_model.fit(
    train_bags, train_bag_labels,
    validation_data=(test_bags, test_bag_labels),
    epochs=10,
    batch_size=16
)

# Step 5: Test with Example Cases
def test_model_with_example_cases(model, test_bags, test_bag_labels):
    predictions = model.predict(test_bags)
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(5):  # Display results for 5 example bags
        print(f"Bag {i+1}: Predicted Label = {CATEGORIES[predicted_labels[i]]}, True Label = {CATEGORIES[test_bag_labels[i]]}")

test_model_with_example_cases(mip_model, test_bags, test_bag_labels)
