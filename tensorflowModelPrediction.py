"""project: creating a model that predicts the accuracy of images using
machine learning (tensorflow). this project uses tensorflow's sample data set/
tensorflow keras, as high level API to build and train models in tensorflow."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt #for graphing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gpu

#load in dataset using keras
data = keras.datasets.fashion_mnist

"""need to split dataset into testing and training data
for machine learning/neural networks, you don't want to pass in all the data when
you train. you want to pass in 80%ish of your data into the network to train and
then test the rest of the data for accuracy"""

#'train' = arrays for the training set, the data that the model uses
#'test' = the data that the model is tested against
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#shrink data down by dividing the pixel value
#we will be using 28 x 28
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    #flatten the data to an entire array of 784 elements instead of 
    #having 28 rows and 28 cols
    keras.layers.Flatten(input_shape=(28,28)), 
    keras.layers.Dense(128, activation="relu"), #"rectify linear unit" - very fast activation function
    keras.layers.Dense(10, activation="softmax") #picks values for neurons that add up to 1
    ])


#tweaking weights and bias
#defining activation functions, etc
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5) #epoch - decides how many times you see an image

#train the model
prediction = model.predict(test_images) #gives predictions by passing in a bunch of items
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#get the largest value and find the index, returning the highest predicted percentage
#get the name of the item by passing it into class_names
#this sets up a way to see the images of what it actually is vs the prediction 
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]]) 
    plt.title("Prediction" + class_names[np.argmax(prediction[i])])
    plt.show()





