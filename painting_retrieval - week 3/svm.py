# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:25:00 2018

@author: Zaius
"""

import numpy as np # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

def train_SVM():
    clg = svm.SVC()
    