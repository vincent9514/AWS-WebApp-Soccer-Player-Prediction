import pickle
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.append('../../../develop/src/data/')
import dataLoading
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import logging


def value_prediction(FIFA):
    """This function predicts a FIFA Players Transfer Market Value for a given input dataset.

    Args:
        FIFA: numpy.ndarray-input arrtibute array
        
    Returns:
        A predict FIFA Player Market Value.
    """
    model_path = './develop/models/rfr.pkl'
    glove_word2vec_path = './develop/data/external/CompleteDataset.csv'
    # Model path inputed
    logging.info('Model path inputed')

    # Clean text string
    cleaned_FIFA = dataLoading.load_data(FIFA)
    logging.info('Clean text string')

    # Load pickled model
    model = pickle.load(open(model_path, 'rb'))
    random_forest_model = pickle.load(model)
    logging.info('Load pickled model')

    # Predict final result
    raw_output= random_forest_model.predict(cleaned_FIFA[2])
    real_output=raw_output[0]

    #generate the output
    logging.info('Output generated')



