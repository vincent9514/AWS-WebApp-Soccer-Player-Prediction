# MSiA423-Vincent-Developer
Developer: Vincent, Product Owner: Rush, QA: Lauren


What is Soccer Player Prediction?
--------------

Predict soccer player’s value by using predictive models based on the features including base stats, skills, preferred position, and club, etc to assist club manager to gain negotiation power in the transfer market, offer wise transfer fee and wage, and ultimately benefit the club from the deal.

Project Charter:
--------------

* Vision: Assist club manager to gain negotiation power in the transfer market, offer wise transfer fee and wage, and ultimately benefit the club from the deal.
* Mission: Predict player’s value, wage, and overall rating by using predictive models based on the features including base stats, skills, preferred position, and club, etc.
* Success criteria: A Root Mean Square Error (RMSE) of lower than $800000 to measure the performance of the transfer market value prediction model and demonstrate the effectiveness of the web application.


If you want to know more, this is a list of selected starting points:

* FIFA18 Complete Dataset on Kaggle. https://www.kaggle.com/thec03u5/fifa-18-demo-player-dataset
* FIFA Introduction. https://www.easports.com/fifa


Getting Started:
--------------

### Set Up Your Virtual Environment

Create a new virtual environment

```
virtualenv -p python3 fifapredict
source fifapredict/bin/activate
```

Install requirements

```
pip install -r requirements.txt
```

### Get FIFA Player Dataset

Download FIFA18 complete dataset from [here]https://www.kaggle.com/thec03u5/fifa-18-demo-player-dataset/downloads/fifa-18-demo-player-dataset.zip/5
Unzip and store in `/data/external/`

### Set up your `.evn` file

* Doing the following format in the config.py:

```
DIALECT = 'mysql'
DRIVER = 'pymysql'
USERNAME = '   '
PASSWORD = '   '
HOST = '   '
PORT = '   '
DATABASE = '   '
SQLALCHEMY_DATABASE_URI = "{}+{}://{}:{}@{}:{}/{}".format(DIALECT,DRIVER,USERNAME,PASSWORD,HOST,PORT,DATABASE)
SQLALCHEMY_TRACK_MODIFICATIONS=False
```

### Run code

```
make all
python app/Flask/application.py
```

Preprocessing Data
--------------

* Value String Cleansing
* Country to Continent
* Potential Points
* Unique Position Name
* Preferred Position Number

EDA
--------------

* Initial Feature Selection
* Top 20 players
* Group Players by Overall
* Age vs Potential Points
* Group by Preferred Positions

Modeling
--------------
* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest
* Neural Network


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │       ├── CompleteDataset.csv
    |       ├── PlayerAttributeData.csv
    |       ├── PlayerPersonalData.csv
    |       └── PlayerPlayingPositionData.csv
    |
    ├── docs               <- A Sphinx project showing the introduction, code, and tutorial
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── train_model.py <- Python Script to train models including linear, ridge, lasso, randomforest, neural network.
    │   ├── lm.pkl         <- The pickle linear regression model.
    │   └── rfr.pkl        <- The pickle randomforest model.
    |
    ├── app                <- Web Application
    │   └── Flask          <- Web Application constructed by Flask platform
    │       ├── application.py
    |       ├── requirements.txt
    |       ├── config.py
    |       ├── static
    |       ├── templates
    |       └── tests
    |           ├── test_basic.py
    |           ├── test_flask.py
    |           ├── test_linear_model.py
    |           ├── test_lasso_model.py
    |           ├── test_ridge_model.py
    |           ├── test_nnet_model.py
    |           └── test_randomforest_model.py
    |
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   |                     the creator's initials, and a short `-` delimited description, e.g.
    │   |                     `1.0-jqp-initial-data-exploration`.
    |   └── EDA.ipynb      <- EDA Notebook to explain the FIFA Dataset.
    |
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.ipynb
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
