# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:33:25 2017

@author: Akhil
"""


#!/usr/bin/python

#===============================================
# image_manip.py
#
# some helpful hints for those of you
# who'll do the final project in Py
#
# bugs to vladimir dot kulyukin at usu dot edu
#===============================================

import argparse
import cv2
import os
import numpy as np
import pickle

# two dictionaries that map integers to images, i.e.,
# 2D numpy array.
TRAIN_IMAGE_DATA = []
TEST_IMAGE_DATA  = []

# the train target is an array of 1's
TRAIN_TARGET = []
# the set target is an array of 0's.
TEST_TARGET  = []

### Global counters for train and test samples
NUM_TRAIN_SAMPLES = 0
NUM_TEST_SAMPLES  = 0

## define the root directory
ROOT_DIR = 'C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/'

## read the single bee train images
YES_BEE_TRAIN = ROOT_DIR + 'single_bee_train'

for root, dirs, files in os.walk(YES_BEE_TRAIN):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA.append(img)
            TRAIN_TARGET.append(int(1))
        NUM_TRAIN_SAMPLES +=1


## read the single bee test images
YES_BEE_TEST = ROOT_DIR + 'single_bee_test'

for root, dirs, files in os.walk(YES_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            # print img.shape
            TEST_IMAGE_DATA.append(img)
            TEST_TARGET.append(int(1))
        NUM_TEST_SAMPLES += 1

## read the no-bee train images
NO_BEE_TRAIN = ROOT_DIR + 'no_bee_train'

for root, dirs, files in os.walk(NO_BEE_TRAIN):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA.append(img)
            TRAIN_TARGET.append(int(0))
        NUM_TRAIN_SAMPLES += 1
        
# read the no-bee test images
NO_BEE_TEST = ROOT_DIR + 'no_bee_test'

for root, dirs, files in os.walk(NO_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TEST_IMAGE_DATA.append(img)
            TEST_TARGET.append(int(0))
        NUM_TEST_SAMPLES += 1

print (NUM_TRAIN_SAMPLES)
print (NUM_TEST_SAMPLES)

TRAIN_IMAGE_DATA = np.asarray(TRAIN_IMAGE_DATA).astype(np.float32)
TEST_IMAGE_DATA = np.asarray(TEST_IMAGE_DATA).astype(np.float32)
TRAIN_TARGET = np.asarray(TRAIN_TARGET).astype(np.float32)
TEST_TARGET = np.asarray(TEST_TARGET).astype(np.float32)

with open("C:/Users/Akhil/OneDrive - aggiemail.usu.edu/Akhil/Fall 2017/TOC/nn_train/output.dat","wb") as f:
    pickle.dump(TRAIN_IMAGE_DATA, f)
    pickle.dump(TEST_IMAGE_DATA, f)
    pickle.dump(TRAIN_TARGET,f)
    pickle.dump(TEST_TARGET,f)
    

######################################