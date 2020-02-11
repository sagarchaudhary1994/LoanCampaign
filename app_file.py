import os
from flask import Flask,jsonify,request
from flask_restful import Api, Resource
from Model_Train import loadData,modelBuild
from DataClean import DataClean
from sklearn.externals import joblib
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np

app=Flask(__name__)
api=Api(app)

if not os.path.isfile('Thera_Model.model'):
    modelBuild()
model=joblib.load('Thera_Model.model')

if not os.path.isfile('Thera_Column.pkl'):
    loadData()
model_columns=joblib.load('Thera_Column.pkl')



class MakePrediction(Resource):
    @staticmethod
    def post():
        json_ = request.json
        query = pd.DataFrame(json_)
        query = DataClean(query)
        prediction = list(model.predict(query))
        return jsonify({"Prediction": str(prediction)})

api.add_resource(MakePrediction,"/predict")
if __name__=='__main__':
    app.run(port=5000,debug=True)




