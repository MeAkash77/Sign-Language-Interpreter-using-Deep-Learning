import numpy as np
import pickle
import cv2
import os
from glob import glob
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Set TensorFlow to use the 'tf' image data format
K.set_image_data_format('channels_last')  # 'channels_last' is the preferred format for TensorFlow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Function to get the size of images
def get_image_size():
    img = cv2.imread('gestures/1/100.jpg', 0)
    if img is None:
        raise FileNotFoundError("The sample image was not found. Please check the file path.")
    return img.shape

# Function to get the number of classes
def get_num_of_classes():
    return len(glob('gestures/*'))

# Get image dimensions
image_x, image_y = get_image_size()

# Define the CNN model
def cnn_model():
    num_of_classes = get_num_of_classes()  # Number of output classes
    model = Sequential()
    
    # Add Conv2D layers and MaxPooling layers
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    
    # Flatten the output and add Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(Dense(num_of_classes, activation='softmax'))  # Output layer with softmax for classification
    
    # Use SGD optimizer with learning rate decay
    sgd = optimizers.SGD(learning_rate=1e-2)
    
    # Compile the model with categorical crossentropy loss for multi-class classification
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Define checkpoint to save the best model during training
    filepath = "cnn_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    
    return model, callbacks_list

# Function to train the model
def train():
    # Load training and validation data
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    # Reshape images to match the input shape of the model
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
    
    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=get_num_of_classes())
    val_labels = to_categorical(val_labels, num_classes=get_num_of_classes())

    print(val_labels.shape)  # Check the shape of validation labels
    
    # Initialize the model and callbacks
    model, callbacks_list = cnn_model()
    
    # Show model summary
    model.summary()
    
    # Train the model
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks=callbacks_list)
    
    # Evaluate the model on validation data
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    
    # Save the trained model
    model.save('cnn_model_keras2.h5')

# Train the model
train()

# Clear the Keras session after training
K.clear_session()

