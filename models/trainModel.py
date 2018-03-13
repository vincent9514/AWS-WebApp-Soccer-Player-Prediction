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





# clean the value string
def str2number(amount):
    """This function makes different dollar units consistent and return cleansed column

    Before returning the cleansed column, it transfers the value with unit "Million"and
    "K"in the column input to the value with dollar unit

    Args:
        amount: input list

    Returns:
        the list with cleased dollar unit
    """
    if amount[-1] == 'M':
        return float(amount[1:-1]) * 1000000
    elif amount[-1] == 'K':
        return float(amount[1:-1]) * 1000
    else:
        return float(amount[1:])


def find_continent(x):
    """This function categorize nations into continent and return continent strings

    continents dictionary was collected through online open source and generated manually

    Args:
        x: input country value

    Returns:
        the value in continent

    """

    continents = {
        'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina', 'Burundi', 'Cameroon', 'Cape Verde',
                   'Central African Republic', 'Chad', 'Comoros', 'Congo', 'DR Congo', 'Djibouti', 'Egypt',
                   'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea Bissau',
                   'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania',
                   'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda',
                   'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa',
                   'South Sudan', 'Sudan', 'Swaziland', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe',
                   'Burkina Faso'],
        'Antarctica': ['Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau',
                       'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu'],
        'Asia': ['Afghanistan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Burma (Myanmar)', 'Cambodia', 'China',
                 'China PR', 'East Timor', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan',
                 'Kazakhstan', 'North Korea', 'South Korea', 'Korea Republic', 'Korea DPR', 'Kuwait', 'Kyrgyzstan',
                 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Nepal', 'Oman', 'Pakistan', 'Palestine',
                 'Philippines', 'Qatar', 'Russian Federation', 'Saudi Arabia', 'Singapore', 'Sri Lanka', 'Syria',
                 'Tajikistan', 'Thailand', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam',
                 'Yemen', 'Russia'],
        'Australia Oceania': ['Australia', 'New Caledonia'],
        'Europe': ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia Herzegovina',
                   'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France',
                   'FYR Macedonia', 'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo',
                   'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Malta', 'Moldova', 'Monaco',
                   'Montenegro', 'Netherlands', 'Northern Ireland', 'Norway', 'Poland', 'Portugal', 'Romania',
                   'San Marino', 'Scotland', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland',
                   'Ukraine', 'England', 'Vatican City', 'Republic of Ireland', 'Wales'],
        'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba',
                          'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras',
                          'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia',
                          'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States'],
        'South America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Curacao', 'Ecuador', 'Guyana',
                          'Paraguay', 'Peru', 'Suriname', 'Trinidad & Tobago', 'Uruguay', 'Venezuela']}
    for key in continents:
        if x in continents[key]:
            return key
    return np.NaN



def load_data(input_path):
    """This function loads a .csv and returns the X_train, X_test, y_train, y_test values for model input

    Before returning the model inputs, it converts the variables into desired format.

    Args:
        input_path (str): Path to datasets

    Returns:
        X_train, X_test, y_train, y_test values for model input
    """
    dataset = pd.read_csv('input_path', header=0)
    interesting_columns = [
        "Photo", 'Name', 'Age', 'Nationality', 'Overall',
        'Potential', 'Club', 'Value', 'Wage', 'Special',
        'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
        'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
        'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',
        'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions',
        'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
        'Positioning', 'Reactions', 'Short passing', 'Shot power',
        'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
        'Strength', 'Vision', 'Volleys',
        'Preferred Positions']
    dataset = pd.DataFrame(dataset, columns=interesting_columns)  # Initial Feature Selection
    dataset['ValueNum'] = dataset['Value'].apply(lambda x: str2number(x))
    dataset['WageNum'] = dataset['Wage'].apply(lambda x: str2number(x))
    dataset['PotentialPoints'] = dataset['Potential'] - dataset['Overall']  # create Potential point
    dataset['Position'] = dataset['Preferred Positions'].str.split().str[0]  # create position
    dataset['PositionNum'] = dataset['Preferred Positions'].apply(lambda x: len(x.split()))  # count position number
    dataset['Continent'] = dataset['Nationality'].apply(lambda x: find_continent(x))

    X = dataset[['Age', 'WageNum', 'Overall',
                 'Potential', 'Special',
                 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
                 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
                 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',
                 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions',
                 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
                 'Positioning', 'Reactions', 'Short passing', 'Shot power',
                 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
                 'Strength', 'Vision', 'Volleys','PositionNum']]

    y = dataset['ValueNum']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


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

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lm = LinearRegression()
    md1 = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)

    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

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
    ridge = Ridge(normalize=True)
    coefs = []
    alphas = 10 ** np.linspace(10, -2, 100) * 0.5
    for a in alphas:
        ridge.set_params(alpha=a)
        ridge.fit(X_train, y_train)
        coefs.append(ridge.coef_)

    ridge2 = Ridge(alpha=4, normalize=True)
    ridge2.fit(X_train, y_train)  # Fit a ridge regression on the training data
    pred2 = ridge2.predict(X_test)  # Use this model to predict the test data
    print(pd.Series(ridge2.coef_, index=X_train.columns))  # Print coefficients
    print(mean_squared_error(y_test, pred2))

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
    # Lasso
    lasso = Lasso(max_iter=10000, normalize=True)
    coefs = []
    alphas = 10 ** np.linspace(10, -2, 100) * 0.5
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(scale(X_train), y_train)
        coefs.append(lasso.coef_)

    lasso2 = Lasso(alpha=1000, normalize=True)
    lasso2.fit(X_train, y_train)  # Fit a lasso regression on the training data
    pred3 = lasso2.predict(X_test)  # Use this model to predict the test data
    print(pd.Series(lasso2.coef_, index=X_train.columns))  # Print coefficients
    print(mean_squared_error(y_test, pred3))

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
    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)
    rfr_pred = rfr.predict(X_test)
    print('MAE_rfr:', metrics.mean_absolute_error(y_test, rfr_pred))
    print('MSE_rfr:', metrics.mean_squared_error(y_test, rfr_pred))
    print('RMSE_rfr:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))

    with open('rfr.pkl', 'wb') as fid:
        pickle.dump(rfr, fid, 2)

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
    mlp = MLPRegressor(hidden_layer_sizes=(30, 30, 30), max_iter=1000)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    print('MAE_nnet:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE_nnet:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE_nnet:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    return mlp, predictions

















