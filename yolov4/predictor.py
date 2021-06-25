# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:32:40 2019

@author: naresh.gangiredd
"""

import os
import json
# from sklearn.externals import joblib
import flask
import boto3
import time
import pyarrow
from pyarrow import feather
#from boto3.s3.connection import S3Connection
#from botocore.exceptions import ClientError
#import pickle
# import modin.pandas as pd
import pandas as pd
from detect_object import *
import logging

#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

logging.info("Model Path" + str(model_path))

# Load the model components
# regressor = joblib.load(os.path.join(model_path, 'Regx.pkl'))
# logging.info("Regressor" + str(regressor))

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    # print('hooooooooooooooooooo')
    # bucket = "itzikbucket18"
    # video_path='test_football.mp4'
    # try:
    #     s3_resource = boto3.resource('s3')
    #     s3_resource.Bucket(bucket).download_file(video_path, video_path)
    #     print('directory contents:',os.listdir())
    #     print(video_path)

    #     # success=detect_object(video_path,bucket = "itzikbucket18",input_size=416)

    # except Exception as e:
    #     traceback.print_exc()

    return flask.Response(response= json.dumps({"result":"pong"}), status=200, mimetype='application/json' )


@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    video_path = input_json['input']['video']
    print('insude detection API:',video_path)
    # predictions = float(regressor.predict([[input]]))
    success=detect_object(video_path,bucket = "itzikbucket18",input_size=416)

    # Transform predictions to JSON
    result = {
        'output': success
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')