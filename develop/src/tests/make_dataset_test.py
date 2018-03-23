import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sys
import os
import pickle
import logging


sys.path.insert(0, os.path.abspath('.'))
sys.path.append('../../..')
from develop.src.data import dataLoading

#readind input dataset
FIFA_path = '../../../develop/data/external/CompleteDataset.csv'


def X_train_load_data_type():
    #check the training x data type
    assert type(dataLoading.load_data(
            FIFA_path)[0]) == numpy.ndarray


def X_test_load_data_type():
    #check the testing x data type
    assert type(dataLoading.load_data(
            FIFA_path)[1]) == numpy.ndarray


def y_train_load_data_type():
    #check the training y data type
    assert type(dataLoading.load_data(
                 FIFA_path)[2]) == numpy.ndarray


def y_test_load_data_type():
    #check the testing y data type
    assert type(dataLoading.load_data(
                 FIFA_path)[3]) == numpy.ndarray


def X_train_load_data_dim():
    #check if the training x data dimension is as same as expected
    assert dataLoading.load_data(FIFA_path)[0].shape == (12586, 40)


def X_test_load_data_dim():
    #check if the testing x data dimension is as same as expected
    assert dataLoading.load_data(FIFA_path)[1].shape== (5395, 40)


def y_train_load_data_dim():
    #check if the training y data dimension is as same as expected
    assert dataLoading.load_data(FIFA_path)[2].shape== (12586,)


def y_test_load_data_dim():
    #check if the testing y data dimension is as same as expected
    assert dataLoading.load_data(FIFA_path)[3].shape== (5395,)

