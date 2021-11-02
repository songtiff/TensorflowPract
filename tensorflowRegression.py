#predict fuel efficiency 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#make numpy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

#get data and import using pandas
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                            na_values='?', comment='\t',
                            sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

#clean dataset
print(dataset.isna().sum())
dataset = dataset.dropna()

#one-hot encode the values in origin column because it is categorical 
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail)

#split data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index) #used at final evaluation 

#inspect data
print(sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde'))
print(train_dataset.describe().transpose())

#separate target value aka 'label' from features- label is the value that
#you will train the model to predict

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#normalize data
print(train_dataset.describe().transpose()[['mean', 'std']])

#add feature normalization 
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

#calculate mean and variance and store it in a layer
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
    print('First example: ', first)
    print()
    print('Normalized: ', normalizer(first).numpy())
