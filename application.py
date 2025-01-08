import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import pandas
from sklearn.preprocessing import StandardScaler
import pickle

application = Flask(__name__)
app = application

# Load the trained model and scaler
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])  
def predict_datapoint():
    if request.method=='POST':
    # Extract features from the form -Temperature	RH	Ws	Rain	FFMC	DMC	ISI	Classes	Region
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws= float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
    
        # Scale the features
        scaled_features = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        
        # Make prediction
        prediction = ridge_model.predict(scaled_features)
        
        return render_template('home.html', result=prediction[0])

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")