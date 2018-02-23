DIALECT = 'mysql'
DRIVER = 'pymysql'
USERNAME = 'vincentwang'
PASSWORD = 'wangziwen199514'
HOST = 'flaskfinal.cjzatzpdiq07.us-east-2.rds.amazonaws.com'
PORT = '3306'
DATABASE = 'flaskfinal'

SQLALCHEMY_DATABASE_URI = "{}+{}://{}:{}@{}:{}/{}".format(DIALECT,DRIVER,USERNAME,PASSWORD,HOST,PORT,DATABASE)

SQLALCHEMY_TRACK_MODIFICATIONS=False
