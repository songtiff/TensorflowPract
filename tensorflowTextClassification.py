"""project: Text classification that decodes integers pointing to words 
to words pointing to integers and determines whether the review is 
good or bad with a limit of 250 words max in a review."""

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
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) 

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
model.add(keras.layers.Embedding(88000, 16)) #groups words in a similar way
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

#define mode
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#check how well model is working
x_val = train_data[:10000] #just get 10000 instead of 25000 entries 
x_train = train_data[10000:]

y_val = train_labels[:10000] 
y_train = train_labels[10000:]

#fit model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)
print(results)

model.save("model.h5") #h5 is an extension for saving model in keras tf

#looks up mapping for all words and returns an encoded list
def review_encode(s):
    encoded = [1] #<START> = 1, setting a starting tag

    for word in s: 
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

model = keras.models.load_model("model.h5")

#load in outside sample data file
with open("testmodel.txt", encoding="utf-8") as f:
    for line in f.readlines(): 
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line) #original text
        print(encode) #encoded review
        print(predict[0]) #whether the model thinks the review is negative or positive




