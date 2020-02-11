import os
import pandas as pd
import numpy as np
from Model_Train import EDA
from sklearn.externals import joblib

##Treating the new data the same way as the train data
def DataClean(data):
    data=data
    columns=data.columns
    model_columns = joblib.load('Thera_Column.pkl')
    if not os.path.isfile("FinalColumns.pkl"):
        EDA()
    FinalColumn=joblib.load("FinalColumns.pkl")
    #miscol=[]
    for col in columns:
        if col not in model_columns.difference(['Personal Loan']):
            #miscol.append(col)
            data[col]=None
    # Renaming the column names.
    data = data.rename(
        columns={"Age (in years)": "Age", "Experience (in years)": "Experience", "Income (in K/month)": "Income",
                 "ZIP Code": "Zip",
                 "Family members": "FamilyMem", "Securities Account": "SecAct",
                 "CD Account": "CDAct"})
    # Changing the datatypes of variables
    data.Zip = data["Zip"].astype('object')
    data.Education == data["Education"].astype('object')
    data.Mortgage = data["Mortgage"].astype('float64')
    data.SecAct = data['SecAct'].astype('object')
    data.CDAct = data['CDAct'].astype('object')
    data.Online = data['Online'].astype('object')
    data.CreditCard = data['CreditCard'].astype('object')
    # Checking the null column in the dataset
    nullcol = data.columns[data.isnull().any()]
    nullcol.ravel()
    # Rows for which value of Family Members is missing
    a = data[data.isnull().any(axis=1)][nullcol].index
    # For now imputing the missing family with fixed value "2"
    data.loc[a, "FamilyMem"] = 2
    data.FamilyMem = data['FamilyMem'].astype('object')

    dropcol = ["ID", "Zip"]
    mydata = data.drop(dropcol, axis=1)
    print(mydata)
    # Changing the value ofcategorical variables to make the interpretation easier later
    mydata['Education'][mydata['Education'] == 1] = "Undergrad"
    mydata['Education'][mydata['Education'] == 2] = "Graduate"
    mydata['Education'][mydata['Education'] == 3] = "Profesional"
    mydata['SecAct'][mydata['SecAct'] == 1] = "Y"
    mydata['SecAct'][mydata['SecAct'] == 0] = "N"
    mydata['CDAct'][mydata['CDAct'] == 1] = "Y"
    mydata['CDAct'][mydata['CDAct'] == 0] = "N"
    mydata['Online'][mydata['Online'] == 1] = "Y"
    mydata['Online'][mydata['Online'] == 0] = "N"
    mydata['CreditCard'][mydata['CreditCard'] == 1] = "Y"
    mydata['CreditCard'][mydata["CreditCard"] == 0] = "N"
    # For categorical values we need to create the dummy variables and drop the forst variable
    mydata = pd.get_dummies(mydata)
    for col in FinalColumn.difference(["PLoan_Y"]):
        if col not in mydata.columns:
            mydata[col]=0
    rm_col=[]
    for col in mydata.columns:
        if col not in FinalColumn:
            rm_col.append(col)
    #print(f'Extra columns in input data {rm_col}')
    #print(f'Columns in final model {FinalColumn}')
    mydata=mydata.drop(rm_col,axis=1)
    return mydata