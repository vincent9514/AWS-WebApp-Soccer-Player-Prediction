#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:59:21 2018

@author: Vincent Wang
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def linear_regression(dataset):
    dt2=dataset[['Age', 'Overall',
       'Potential', 'WageNum', 'Special',
       'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',
       'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions',
       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
       'Positioning', 'Reactions', 'Short passing', 'Shot power',
       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
       'Strength', 'Vision', 'Volleys',
       'PositionNum','ValueNum']]
    dt2[dt2.columns] = dt2[dt2.columns].apply(pd.to_numeric, errors='coerce')
    X = dt2[['Age', 'Overall',
       'Potential', 'WageNum', 'Special',
       'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',
       'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions',
       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
       'Positioning', 'Reactions', 'Short passing', 'Shot power',
       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
       'Strength', 'Vision', 'Volleys',
       'PositionNum']]
    y = dt2['ValueNum']
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    X_1 = pd.DataFrame(X)
    X_1.columns = ['Age', 'Overall',
       'Potential', 'WageNum', 'Special',
       'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',
       'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions',
       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
       'Positioning', 'Reactions', 'Short passing', 'Shot power',
       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
       'Strength', 'Vision', 'Volleys',
       'PositionNum']
    X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.4, random_state=101)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lm = LinearRegression()
    md1=lm.fit(X_train,y_train)
    #Model Evaluation
    coeff_df = pd.DataFrame(lm.coef_,X_1.columns,columns=['Coefficient'])
    predictions = lm.predict(X_test)
    
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
   
    return md1



def redge_regression(X_train, X_test, y_train, y_test):
    #ridge
    ridge = Ridge(normalize=True)
    coefs = []
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)
        
    ridge2 = Ridge(alpha=4, normalize=True)
    ridge2.fit(X_train, y_train) # Fit a ridge regression on the training data 
    pred2 = ridge2.predict(X_test) # Use this model to predict the test data 
    print(pd.Series(ridge2.coef_, index=X_1.columns)) # Print coefficients
    print(mean_squared_error(y_test, pred2))
    
    return ridge2,pred2
    


def lasso_regression(X_train, X_test, y_train, y_test):
    #Lasso
    lasso = Lasso(max_iter=10000, normalize=True)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(scale(X_train), y_train)
        coefs.append(lasso.coef_)
        
    lasso2 = Lasso(alpha=1000, normalize=True)
    lasso2.fit(X_train, y_train) # Fit a lasso regression on the training data 
    pred3 = lasso2.predict(X_test) # Use this model to predict the test data 
    print(pd.Series(lasso2.coef_, index=X_1.columns)) # Print coefficients
    print(mean_squared_error(y_test, pred2))
    
    return lasso, pred3