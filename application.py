import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template
## render_template - To find out the Url of the HTML file 

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle 
ridge_model = pickle.load(open('models /ridge.pkl','rb'))
standard_scaler = pickle.load(open('models /scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature')) # Temperature inside request.form.get should be matching the input taken in home.html page
        RH = float(request.form.get('RH')) # RH inside request.form.get should be matching the input taken in home.html page
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])
    else : 
        return render_template('home.html')

## difference between get() and post() methods
## get - e.g. opening google.com
## post - e.g. opening any site inside google.com like opening geeksforgeeks inside google.com

if __name__=="__main__":
    app.run(host="0.0.0.0") ## To change the port number,you can give port='8080' or any other port number 
