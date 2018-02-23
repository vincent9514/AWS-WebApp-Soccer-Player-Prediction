
from flask import Flask,request,render_template
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import config

app = Flask(__name__,static_url_path='/static')
app.config.from_object(config) #connect db

db= SQLAlchemy(app)

#creat a table in db
#class Article(db.Model):
#    __tablename__='article'
#    id=db.Column(db.Integer,primary_key=True,autoincrement=True)
#    title=db.Column(db.String(100),nullable=False)

#db.create_all()#check if connected with db


class Player(db.Model):
    __tablename__='player'
    id=db.Column(db.Integer,primary_key=True,autoincrement=True)
    Age=db.Column(db.Integer)
    Wage=db.Column(db.Integer)
    Overall_Rating=db.Column(db.Integer)
    Potential=db.Column(db.Integer)
    Composure=db.Column(db.Integer)
    Marking=db.Column(db.Integer)
    Reactions=db.Column(db.Integer)
    Vision=db.Column(db.Integer)
    Volleys=db.Column(db.Integer)
    Num_Positions=db.Column(db.Integer)
    
db.create_all()

@app.route('/', methods = ['GET'])
def index():
    #article1=Article(title='aaa') #input data to db
    #db.session.add(article1)
    #db.session.commit()
    context={
            'username':u'Vincent',
            'gender':u'Male',
            'age':u'Age'
            }
    return render_template('index.html')
#render_template('index.html',age=u'100')
#**context


@app.route('/', methods = ['POST'])
def model():
    
    Age = request.form['Age'] 
    Wage = request.form['Wage']
    Overall_Rating = request.form['Overall_Rating']
    Potential = request.form['Potential']
    Composure = request.form['Composure']
    Marking = request.form['Marking']
    Reactions = request.form['Reactions']
    Vision = request.form['Vision']
    Volleys = request.form['Volleys']
    Num_Positions = request.form['Num_Positions']
    
    player1=Player(Age=Age,Wage=Wage,Overall_Rating=Overall_Rating,Potential=Potential,Composure=Composure,
                   Marking=Marking,Reactions=Reactions,Vision=Vision,Volleys=Volleys,Num_Positions=Num_Positions)
    db.session.add(player1)
    db.session.commit()
    #return render_template('index.html',Age=Age)
    
    
    #SET MODEL INPUT
    input_array=np.ndarray(shape=(1,10), dtype=float, order='F')
    input_array[0][0]=Age #age
    input_array[0][1]=Wage #Wage
    input_array[0][2]=Overall_Rating #Overall_Rating
    input_array[0][3]=Potential #Potential
    input_array[0][4]=Composure #Composure
    input_array[0][5]=Marking #Marking
    input_array[0][6]=Reactions #Reactions
    input_array[0][7]=Vision #Vision
    input_array[0][8]=Volleys #Volleys
    input_array[0][9]=Num_Positions #Num_Positions
    
    
    pkl_file = open('lm.pkl', 'rb')
    linearmodel = pickle.load(pkl_file)
    raw_output= linearmodel.predict(input_array)
    real_output=raw_output[0]
    
    return render_template('result.html',prediction=real_output)
    
#    if (Age == 'Age'or Wage == 'Wage' ):
#        return '<h3> test succeed</h3>'
#    else:
#        return '<h3> invalid_Age </h3>'
    


if __name__ == '__main__':
	app.run(port=4883,debug=True)