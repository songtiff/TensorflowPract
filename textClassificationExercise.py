"""This notebook trains a sentiment analysis model to classify movie reviews as positive or negative, 
based on the text of the review. This is an example of binary—or two-class—classification, an important 
and widely applicable kind of machine learning problem"""

import matplotlib.pyplot as plt
import os

from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.gen_string_ops import regex_replace
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gpu
import re
import shutil 
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

#dl and extract dataset from IMDB database
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar = True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read)

#remove additional folders
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

#pass in dataset to train model 
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2, #create a validation split of 80:20
    subset = 'training',
    seed = seed
)

#iterate over dataset and print out examples
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

#validation and test dataset w/ remaining 5000 reviews from training set
#validation set
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'validation', 
    seed = seed
)

#test set
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test',
    batch_size = batch_size
)

#custom standardization function to remove HTML
#default standardizer does not remove HTML tags
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

max_features = 10000 #max cap
sequence_length = 250 #truncate cap

#standardize, tokenize, and vectorize data
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

#fit preprocessing layer to dataset - builds an index of strings to integers
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

#retrieve a batch of 32 reviews and labels from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review ", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review ", vectorize_text(first_review, first_label))

#check string version of each token
print("1287 ", vectorize_layer.get_vocabulary()[1287])
print("313 ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

#apply TextVectorization to train, validation, and test dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

#configure dataset for performance
#ensure data does not bottleneck
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#create model/neural network
embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features+1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.summary()

#loss function and optimizer
model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

#train model    
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

#evaluate model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

#plot of accuracy and loss over time