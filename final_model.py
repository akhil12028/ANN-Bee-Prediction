from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import os
import cv2
import numpy as np

def build():
    bees = input_data(shape=[None, 32, 32, 3], name='input')
    bees = conv_2d(bees, 32, 5, activation='relu', regularizer="L2")
    bees = max_pool_2d(bees, 2)
    bees = local_response_normalization(bees)
    bees = fully_connected(bees, 2, activation='softmax')
    bees = regression(bees,learning_rate=0.001,name='target')
    
    classifier = tflearn.DNN(bees)
    return classifier


def load(dirpath):
    test_data = []
    for root, dirs, files in os.walk(dirpath):
        for item in files:
            if item.endswith('.png'):
                ip = os.path.join(root, item)
                img = (cv2.imread(ip)/float(255))
                test_data.append(img)
    return test_data

def predict(netpath,test_data):
    test_data = np.asarray(test_data).astype(np.float32)
    classifier = build()
    classifier.load(netpath)  
    score = classifier.predict({'input': test_data})
    print("Total accuracy is",score[0]*100)
    return score

def testNet(netpath,dirpath):
    test_data = load(dirpath)
    logits = predict(netpath,test_data)
    count = 0
    for label in logits:
        if label[0]<0.5:
            print("Single bee Image")
            count = count+1
        else:
            print("No bee Image")
    return count/len(test_data)


# Evaluating Accuracy of single bees
#netpath = 'C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/bees_final3.model'
#dirpath = "C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/single_bee_test"
#print(testNet(netpath,dirpath))

# Evaluating Accuracy of no bees
#netpath = 'C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/bees_final3.model'
#dirpath = "C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/no_bee_test"
#print(1-testNet(netpath,dirpath))

#Input netpath and dirpath below

netpath = 'C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/bees_final3.model'
dirpath = "C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/single_bee_test"
testNet(netpath,dirpath)
