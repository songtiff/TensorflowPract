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

#can load our data like this thanks to keras
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

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc:", test_acc)

