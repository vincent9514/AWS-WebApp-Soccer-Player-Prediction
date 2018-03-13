#Lasso regression
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import sys
import os
import pickle
import logging






def test_lasso_model():
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
    assert lasso.predict(inputs)==np.array([3937000.0,19570000.0,49285000.0,25395000.0,3422000.0])

