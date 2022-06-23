from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
import os
import numpy as np
import pickle
import time

import pandas as pd
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from LoanPrediction import * 
from sklearn import linear_model
app = Flask(__name__, static_folder='static')

@app.route("/")
def index():
    return render_template('home.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    Gender = str(request.form['Gender'])
    Married = str(request.form['Married'])
    Dependents = str(request.form['Dependents'])
    Education = str(request.form['Education'])
    Self_Employed = str(request.form['Self_Employed'])
    ApplicantIncome = str(request.form['ApplicantIncome'])
    CoapplicantIncome = str(request.form['CoapplicantIncome'])
    LoanAmount = str(request.form['LoanAmount'])
    Loan_Amount_Term = str(request.form['Loan_Amount_Term'])
    Credit_History = str(request.form['Credit_History'])
    Property_Area = str(request.form['Property_Area'])
    col_names =  ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
    attendance = pd.DataFrame(columns = col_names)
    attendance.loc[len(attendance)] = [Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]
    fileName="inputdata\inputdata.csv"
    attendance.to_csv(fileName,index=False)
    data=msg_process1()
    if data == 1:
        first="Loan application accepted"
    else:
        first=" Loan application Rejected"
     
    
    
    return render_template('Result.htm', first=str(first))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
