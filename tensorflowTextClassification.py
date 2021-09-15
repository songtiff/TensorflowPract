"""project: """

import tensorflow as tf
from tensorflow import keras 
import numpy as np
from tensorflow.python.ops.gen_array_ops import reverse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gpu

#load in movie dataset  
data = keras.datasets.imdb 

#since this dataset contains a bunch of words, we want the 10000th most frequent words
#prints out integer encoded words - each integer stands for a word
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000) 

word_index = data.get_word_index() #gives us a tuple
word_index = {k : (v + 3) for k, v in word_index.items()} #break the tuple into key and value pairings k = word, v = integer

#allows personal value assigning
word_index["<PAD>"] = 0 #assign as a max limit to the amount of words that can be used in a review
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#swap values in the keys
#we currently have words pointing to integers, but we reverse to get the words pointing to the integers
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 

#redefine testing data 
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text]) #try to get index i, otherwise return a ?

#model 
#final output expectation : whether the review is good or bad
#neuron output will be either 0 or 1 to give us a probability where a review is a certain percentage either 0 or 1
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()