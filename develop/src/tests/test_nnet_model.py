#neural_network
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import sys
import os
import pickle
import logging

def test_nnet_model():
    #user inputs
    user_inputs = {
        'Age': [20,25,28,30,35],
        'Wage':[10000,50000,150000,100000,60000],
        'Overall_Rating':[70,80,90,85,78],
        'Potential':[90,85,80, 75,70],
        'Composure':[80,85,90,95,98],
        'Marking':[30,50,70,80,20],
        'Reactions':[90,95,92,85,75],
        'Vision':[70,80,90,95,88],
        'Volleys':[80,85,90,80,75],
        'Num_Positions':[3,3,2,1,1]
    }
    inputs=pf.DataFrame(data=user_inputs)
    
    #check type
    assert isinstance(user_inputs,pd.DataFrame)
    
    #check expected outcome
    assert mlp.predict(inputs)==np.array([4143000.0,21530000.0,42365000.0,30115000.0,3596000.0])

