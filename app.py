import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
ridge_model=pickle.load(open('model/linreg.pkl','rb'))
standard_scaler=pickle.load(open('model/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def bin():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        age=int(request.form.get('Age'))
        bmi = float(request.form.get('BMI'))
        children = int(request.form.get('CHILDREN'))
        smoker = int(request.form.get('SMOKER'))
        region = int(request.form.get('REGION'))
        

        new_data_scaled=standard_scaler.transform([[age,bmi,children,smoker,region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('index.html',result=result[0])

    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
