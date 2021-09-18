"""project: creating a model that predicts the accuracy of images using
machine learning (tensorflow). this project uses tensorflow's sample data set/
tensorflow keras, as a high level API to build and train models in tensorflow."""

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
"""for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]]) 
    plt.title("Prediction" + class_names[np.argmax(prediction[i])])
    plt.show()"""

#graph to look at the full set of 10 class predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
                                            
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#verifying the predictions of several images
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, prediction[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, prediction[i], test_labels)
plt.tight_layout()
plt.show()

#grabbing an image from the dataset to make a prediction on
img = test_images[1]
img = (np.expand_dims(img, 0)) #tf.keras models make predictions on a batch/collection
print(img.shape)

#predict correct label for this image
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45)
plt.show()








