import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint



def msg_process1():

    train=pd.read_csv(r'train.csv')
    train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})
    train.isnull().sum()
    Loan_status=train.Loan_Status
    train.drop('Loan_Status',axis=1,inplace=True)
    test=pd.read_csv('inputdata/inputdata.csv')
    #Loan_ID=test.Loan_ID
    data=train.append(test)
    data.head()
    data.describe()
    data.isnull().sum()

    data.Dependents.dtypes
   

    corrmat=data.corr()
    
    data.Gender=data.Gender.map({'Male':1,'Female':0})
    data.Gender.value_counts()
    corrmat=data.corr()
    
    data.Married=data.Married.map({'Yes':1,'No':0})
    data.Married.value_counts()
    data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
    data.Dependents.value_counts()
    corrmat=data.corr()
     

    data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})
    data.Education.value_counts()
    data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})
    data.Self_Employed.value_counts()
    data.Property_Area.value_counts()
    data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
    data.Property_Area.value_counts()

    corrmat=data.corr()    
    data.Credit_History.size
    data.Credit_History.fillna(np.random.randint(0,2),inplace=True)
    data.isnull().sum()
    data.Married.fillna(np.random.randint(0,2),inplace=True)
    data.isnull().sum()
    data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)
    data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)
    data.isnull().sum()
    data.Gender.value_counts()     
    data.Gender.fillna(np.random.randint(0,2),inplace=True)
    data.Gender.value_counts()
    data.Dependents.fillna(data.Dependents.median(),inplace=True)
    data.isnull().sum()
    corrmat=data.corr()
     
     
    data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)
    data.isnull().sum()
    data.head()
    data.drop('Loan_ID',inplace=True,axis=1)
    data.isnull().sum()
    train_X=data.iloc[:614,]
    train_y=Loan_status
    X_test=data.iloc[614:,]
    seed=7


    from sklearn.model_selection import train_test_split
    train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=seed)

     
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    clf=LogisticRegression()
    clf.fit(train_X,train_y)
     
    df_output=pd.DataFrame()
    #outp=clf.predict(inputdata).astype(int)
    outp=clf.predict(X_test).astype(int)
    print("ddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
    print(X_test)
    print(outp)
    return outp