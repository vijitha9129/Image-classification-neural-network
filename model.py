'''
Created on 

@author: Vijitha Kanumuru

source:


'''

from flask import Flask, request
import pandas as pd 
import numpy as np 
import glob as glob 
import os
import cv2
import matplotlib.pylab as plt 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle


#PICKLE_PATH = "model.pkl"
app = Flask(__name__)


bed = glob.glob('data/Bed/*.jpg')
chair = glob.glob('data/Chair/*.jpg')
sofa = glob.glob('data/Sofa/*.jpg')

new_size = (224, 224)

input_dir1 = 'data/Bed'
output_dir1 = 'resize/resize_bed'
input_dir2 = 'data/Chair'
output_dir2 = 'resize/resize_chair'
input_dir3= 'data/Sofa'
output_dir3 = 'resize/resize_sofa'


##loaded_model = pickle.load(open(PICKLE_PATH,'rb'))

#def save_model(model):
    
    ### Create a Pickle file using serialization 
 #   pickle_out = open(PICKLE_PATH, "wb")
 #   pickle.dump(model, pickle_out)
 #   pickle_out.close()




def resize_img():
    for file_name in os.listdir(input_dir1):
        image = cv2.imread(os.path.join(input_dir1, file_name))
        resized_image = cv2.resize(image, new_size)
        cv2.imwrite(os.path.join(output_dir1, file_name), resized_image)

    for file_name in os.listdir(input_dir2):
        image = cv2.imread(os.path.join(input_dir2, file_name))
        resized_image = cv2.resize(image, new_size)
        cv2.imwrite(os.path.join(output_dir2, file_name), resized_image)
    
    for file_name in os.listdir(input_dir3):
        image = cv2.imread(os.path.join(input_dir3, file_name))
        resized_image = cv2.resize(image, new_size)
        cv2.imwrite(os.path.join(output_dir3, file_name), resized_image)

def train_model():
    
    batch_size = 32
    epochs = 10
    input_shape = (224, 224, 3)
    labels = 3

# Define the model
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(labels, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


    # Set the training and validation data generators
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        'resize',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

   
    val_generator = train_datagen.flow_from_directory(
        'resize',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    # Train the model
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

    ##save_model(model)
   ## model.save_weights('furniture_classification.h5')
    tf.keras.models.save_model(model,"furniture.h5")

    # Plot the training and validation accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == '__main__':
    resize_img()
    train_model()

    model = load_model("furniture.h5")

    img_path = '/home/vijitha/tact/fulhaus-assignment-vijitha/resize/resize_bed/140 cm double bed.jpg'    # dog

    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)
