from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle, to_categorical
import pickle

with open("C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/output.dat","rb") as f:
    training_data = pickle.load(f)
    testing_data = pickle.load(f)
    training_labels = pickle.load(f)
    testing_labels = pickle.load(f)

training_data,training_labels = shuffle(training_data,training_labels)
training_labels = to_categorical(training_labels,2)
testing_labels = to_categorical(testing_labels,2)

def build():
    bees = input_data(shape=[None, 32, 32, 3], name='input')
    bees = conv_2d(bees, 32, 5, activation='relu', regularizer="L2")
    bees = max_pool_2d(bees, 2)
    bees = local_response_normalization(bees)
    bees = fully_connected(bees, 2, activation='softmax')
    bees = regression(bees,learning_rate=0.001,name='target')
    
    classifier = tflearn.DNN(bees)
    return classifier

def save(classifier):
    classifier.save('C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/bees_final3.model')
    
classifier = build()
classifier.fit({'input': training_data}, {'target': training_labels}, n_epoch=100, show_metric=True)
save(classifier)

