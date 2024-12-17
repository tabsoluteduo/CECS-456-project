# CECS 456 project
 
# Running the Medical Image Classification Model

This guide provides step-by-step instructions on how to run the deep learning model for classifying medical images using the Medical MNIST dataset.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python (version 3.6 or higher)
- TensorFlow (version 2.x)
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

You can install the required packages using pip:

bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn

## Downloading the Dataset

1. Download the Medical MNIST dataset from the appropriate source.
2. Unzip the dataset to a directory on your local machine.

## Modifying the File Path

In the code, you will need to specify the path to the directory where you downloaded the Medical MNIST dataset. Open the `myModel.py` file and locate the following line:

python
DATASET_DIR = '(Change according to path in own device)' # Path to unzipped Medical MNIST folder

### Change the File Path

Replace the existing path with the path to your dataset. For example, if you downloaded the dataset to `C:\Datasets\Medical_MNIST`, you would modify the line as follows:

python
DATASET_DIR = 'C:\Datasets\Medical_MNIST' # Update this path to where you downloaded your dataset

Make sure to use double backslashes (`\\`) in the file path to avoid issues with escape characters in Python strings.

## Running the Code

Once you have modified the file path, you can run the model by executing the following command in your terminal or command prompt:

myModel.py or model.py for either DL model

### Expected Output

The program will output the following:

1. A message indicating that the dataset is being loaded.
2. The number of images loaded from the dataset.
3. Training progress, including loss and accuracy metrics for each epoch.
4. Evaluation results on the test set, including test accuracy.
5. Visualizations of random test images along with their true and predicted labels.

## Conclusion

By following these steps, you should be able to successfully run the medical image classification model. If you encounter any issues, ensure that all dependencies are installed and that the dataset path is correctly specified. Happy coding!