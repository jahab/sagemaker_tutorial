try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from io import BytesIO
import json
import sys, os, base64, datetime, hashlib, hmac
from chalice import Chalice
from chalice import NotFoundError, BadRequestError

import sys, os, base64, datetime, hashlib, hmac

app = Chalice(app_name='yolov4-deepsort-api')
app.debug = True

import boto3

sagemaker = boto3.client('sagemaker-runtime')


@app.route('/')
def index():
    return{'hello':'world'}

@app.route('/', methods=['POST'], content_types=['application/json'], cors=True)
def handle_data():
    # Get the json from the request
    input_json = app.current_request.json_body
    # Send everything to the Sagemaker endpoint
    res = sagemaker.invoke_endpoint(
        EndpointName='yolov4-deepsort-ep7',
        Body=input_json,
        ContentType='application/json',
        Accept='Accept'
    )
    return res['Body'].read()
