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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pickle
import logging



def linear_regression(X_train, X_test, y_train, y_test):
    """This function trains the linear regression model.

    Model development was done in a Jupyter notebook and chosen with
    cross-validation accuracy as the performance metric.

    Args:
        X_train: X_train set
        X_test: X_test set
        y_train: y_train set
        y_test: y_test set

    Returns:
        md1: sklearn linear regression model
        predictions: prediction result

    """
    # loading the linear regression function
    logger = logging.getLogger(__name__)
    logger.info('Linear Regression')

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Transforming the input data
    logging.info('Transforming the input data')
    lm = LinearRegression()
    md1 = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    
    # Creating the model and making prediction
    logging.info('Model and prediction')
    return md1,predictions


def ridge_regression(X_train, X_test, y_train, y_test):
    """This function trains the ridge regression model.

    Model development was done in a Jupyter notebook and chosen with
    cross-validation accuracy as the performance metric.

    Args:
        X_train: X_train set
        X_test: X_test set
        y_train: y_train set
        y_test: y_test set

    Returns:
        ridge2: ridge regression model
        pred2: prediction result

    """
    # loading the ridge regression function
    logger = logging.getLogger(__name__)
    logger.info('Ridge Regression')
    ridge = Ridge(normalize=True)
    coefs = []
    alphas = 10 ** np.linspace(10, -2, 100) * 0.5
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X_train, y_train)
        coefs.append(ridge.coef_)

    # Fit a ridge regression on the training data
    ridge2 = Ridge(alpha=4, normalize=True)
    ridge2.fit(X_train, y_train)
    logging.info('Model created')

    # Use this model to predict the test data
    pred2 = ridge2.predict(X_test)
    logging.info('prediction created')

    return ridge2, pred2


def lasso_regression(X_train, X_test, y_train, y_test):
    """This function trains the lasso_regression model.

    Model development was done in a Jupyter notebook and chosen with
    cross-validation accuracy as the performance metric.

    Args:
        X_train: X_train set
        X_test: X_test set
        y_train: y_train set
        y_test: y_test set

    Returns:
        lasso: lasso regression model
        pred3: prediction result

    """
    # loading the lasso regression function
    logger = logging.getLogger(__name__)
    logger.info('Lasso Regression')
    lasso = Lasso(max_iter=10000, normalize=True)
    coefs = []
    alphas = 10 ** np.linspace(10, -2, 100) * 0.5
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(scale(X_train), y_train)
        coefs.append(lasso.coef_)

    # Fit a lasso regression on the training data
    lasso2 = Lasso(alpha=1000, normalize=True)
    lasso2.fit(X_train, y_train)
    logging.info('Model created')

    # Use this model to predict the test data
    pred3 = lasso2.predict(X_test)
    logging.info('prediction created')

    return lasso, pred3


def randomforest(X_train, X_test, y_train, y_test):
    """This function trains the random forest model and pickles the model for later user.

    Model development was done in a Jupyter notebook and chosen with
    cross-validation accuracy as the performance metric.

    Args:
        X_train: X_train set
        X_test: X_test set
        y_train: y_train set
        y_test: y_test set

    Returns:
        rfr: random forest model
        rfr_pred: prediction result

    """
    # loading the random forest function
    logger = logging.getLogger(__name__)
    logger.info('Random Forest')
    
    # Fit random forest model
    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)
    logging.info('Model created')
    
    # Make prediction
    rfr_pred = rfr.predict(X_test)
    logging.info('prediction created')

    with open('rfr.pkl', 'wb') as fid:
        pickle.dump(rfr, fid, 2)
    #pickle the model
    logging.info('Pickle Created')

    return rfr, rfr_pred


def nnet(X_train, X_test, y_train, y_test):
    """

    Args:
        X_train: X_train set
        X_test: X_test set
        y_train: y_train set
        y_test: y_test set

    Returns:
        mlp: neural network model
        predictions: prediction result

    """
    # loading the neural network function
    logger = logging.getLogger(__name__)
    logger.info('Neural Network')
    
    # Fit nnet model
    mlp = MLPRegressor(hidden_layer_sizes=(30, 30, 30), max_iter=1000)
    mlp.fit(X_train, y_train)
    logging.info('Model created')
    
    # Make prediction
    predictions = mlp.predict(X_test)
    logging.info('Prediction Created')

    return mlp, predictions

















