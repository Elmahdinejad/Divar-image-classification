# Divar-image-classification

## Overview

In this project, I've developed an image classification model that utilizes web scraping to gather images from the Divar website and create a dataset with 11 classes. The main goal of this project is to develop a model that can automatically recognize and classify images into one of the specified 11 classes. Such models have applications in various industries, including medical image analysis, object recognition, surveillance systems, augmented reality applications, and media image identification.

## Dataset
The dataset consists of approximately 12,000 images that have been extracted from the Divar website (https://divar.ir) using web scraping and categorized into 11 classes named 206, bird, carpet, cat, cellphone, dog, fish, horse, laptop, perfume, sheep.

## Data Preprocessing
Loading Data from Folders
Images from all directories in the dataset are loaded. The images are then converted from various formats to NumPy arrays.

## Resizing Images
Each image is resized to (64, 64) to ensure uniform dimensions for input into the model.

## Normalizing Pixels
The images are divided by 255 so that pixel values fall within the range of [0, 1], which helps improve model performance.

## Label Encoding
The labels of the images, which correspond to the relevant folder names, are converted to binary numerical form using LabelBinarizer. Each label is transformed into a binary vector with a length equal to the number of classes, where there is only one value of 1 for each sample.

## Splitting Data into Training and Testing Sets
The data is split using train_test_split at an 80% ratio for training and 20% for testing. These preprocessing steps help the model effectively utilize the input data.

## Model Construction
To construct the model, a Convolutional Neural Network (CNN) is used. The model is hierarchically composed of multiple layers, each contributing to improved performance. The layers are defined as follows:

Convolutional Layers: Initially, four convolutional layers are added to the model, each utilizing filters of size (3, 3) to extract various image features. Each convolutional layer uses the ReLU activation function, allowing the model to learn more complex features.

Pooling Layers: After each convolutional layer, a MaxPool layer is added to reduce the size of the feature maps, making the model more efficient.

Batch Normalization: Normalization layers are used to adjust the data and prevent issues arising from high variance in the data.

Dropout: To prevent overfitting, Dropout layers are used to randomly disconnect a percentage of connections during training.

Dense Layers: After the feature extraction stages, the data is connected to a Dense layer that also uses the ReLU activation function. Finally, another Dense layer with a softmax activation function is added to produce the model's final output, indicating the number of classes.

## Model Training
The constructed model is utilized for training following the preparation of data (previously discussed in the preprocessing section). The training process involves the following configurations:

## Defining Training Parameters
The training data (x_train, y_train) and testing data (x_test, y_test) are employed for training and evaluating the model. The number of epochs is set to 20, and the batch size is set to 32.

## Error Reduction (Loss)
The categorical_crossentropy function is used to calculate the error of the model at each prediction. The model is trained using the Adam optimization algorithm.

## Monitoring Model Performance
During training, accuracy and loss are displayed based on the testing data to prevent overfitting.

## Displaying Results
Different charts are utilized to evaluate the model's performance, showing accuracy and loss over various training epochs. These charts help us understand whether the model is learning or experiencing overfitting.

## Saving the Model
After the training is complete, the trained model is saved in a file with the Keras format for future use, allowing it to be employed for subsequent predictions.

## Conclusion
In this project, a Convolutional Neural Network (CNN) model was implemented and trained for image classification from a dataset containing 11 different classes. The goal of this project was to accurately predict and categorize images into their respective classes using advanced deep learning techniques. The results obtained indicate that the model achieved significant accuracy in simulating correct behaviors for image classification.

Throughout the training process, techniques such as Dropout and Batch Normalization were utilized to prevent overfitting and enhance accuracy. Ultimately, the model learned complex features of the images and demonstrated high accuracy in simulating predictions on testing data. By employing techniques like standard feature normalization and training the model with varying epochs, the model's performance was significantly improved.

This project showcases how complex image classification challenges can be addressed using CNNs, leveraging extracted features for more precise predictions. Furthermore, the training and data preprocessing processes clearly illustrate how deep learning models can be optimized for various datasets.
