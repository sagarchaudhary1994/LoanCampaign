#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

#Libraries to Run Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import eli5 #for permutation importance
from eli5.sklearn import PermutationImportance
#import shap #for shap values
from pdpbox import pdp,info_plots # for partial dependency plots
np.random.seed=123
pd.options.mode.chained_assignment=None #ignore any warnings from pandas

def loadData():
    ##Loading Data File
    data = pd.read_csv("TheraBanks.csv")
    data_columns = data.columns
    # Saving the original columns data
    joblib.dump(data_columns.difference(["ID"]), 'Thera_Column.pkl')
    print(f'Columns considered in model are {data_columns}')
    return data

#Defining Exploratory Data Analysis for Data
def EDA():
    data = loadData()
    # Renaming the column names.
    data = data.rename(
        columns={"Age (in years)": "Age", "Experience (in years)": "Experience", "Income (in K/month)": "Income",
                 "ZIP Code": "Zip",
                 "Family members": "FamilyMem", "Personal Loan": "PLoan", "Securities Account": "SecAct",
                 "CD Account": "CDAct"})
    # Changing the datatypes of variables
    data.Zip = data["Zip"].astype('object')
    data.Education == data["Education"].astype('object')
    data.Mortgage = data["Mortgage"].astype('float64')
    data.PLoan = data["PLoan"].astype('object')
    data.SecAct = data['SecAct'].astype('object')
    data.CDAct = data['CDAct'].astype('object')
    data.Online = data['Online'].astype('object')
    data.CreditCard = data['CreditCard'].astype('object')
    # Checking the null column in the dataset
    nullcol = data.columns[data.isnull().any()]
    nullcol.ravel()
    # Rows for which value of Family Members is missing
    a = data[data.isnull().any(axis=1)][nullcol].index
    data.loc[a]
    # For now imputing the missing family with fixed value "2"
    data.loc[a, "FamilyMem"] = 2
    data.FamilyMem = data['FamilyMem'].astype('object')

    dropcol = ["ID", "Zip"]
    mydata = data.drop(dropcol, axis=1)
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
    mydata['PLoan'][mydata['PLoan'] == 1] = "Y"
    mydata['PLoan'][mydata['PLoan'] == 0] = "N"

    # For categorical values we need to create the dummy variables and drop the forst variable
    mydata = pd.get_dummies(mydata, drop_first=True)
    joblib.dump(mydata.columns,"FinalColumns.pkl")
    return mydata

##Method to build the RandomForestClassifier
def modelBuild():
    mydata = EDA()
    # Spitting the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(mydata.drop('PLoan_Y', axis=1), mydata['PLoan_Y'],
                                                        test_size=.3, random_state=10)  ##Split the data
    # print(f'Shape of train data: X:{X_train.shape} and y:{y_train.shape}\nShapeof test data: X:{X_test.shape} and y:{y_test.shape}')

    # Building a Random Forest Model and fitting the model
    model = RandomForestClassifier(n_estimators=500,
                                   random_state=10,
                                   min_samples_split=10,
                                   max_features="sqrt").fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    joblib.dump(model, "Thera_Model.model")
    print(f'Model training finished with accuracy of {accuracy}')







