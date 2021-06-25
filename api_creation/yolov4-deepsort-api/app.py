from io import BytesIO
import json
import sys, os, base64, datetime, hashlib, hmac
from chalice import Chalice
from chalice import NotFoundError, BadRequestError

import sys, os, base64, datetime, hashlib, hmac
import boto3


app = Chalice(app_name='yolov4-deepsort-api')
app.debug = True


sagemaker = boto3.client('sagemaker-runtime')


#invoke AWS runtime endpoint on the command line to test before running the api
# aws sagemaker-runtime invoke-endpoint --endpoint-name yolov4-deepsort-ep7 --cli-binary-format raw-in-base64-out --body '{"input":{"video":"test_football.mp4"}}'   --content-type application/json   --accept application/json results

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


# The view function above will return {"hello": "world"}
# whenever you make an HTTP GET request to '/'.
#
# Here are a few more examples:
#
# @app.route('/hello/{name}')
# def hello_name(name):
#    # '/hello/james' -> {"hello": "james"}
#    return {'hello': name}
#
# @app.route('/users', methods=['POST'])
# def create_user():
#     # This is the JSON body the user sent in their POST request.
#     user_as_json = app.current_request.json_body
#     # We'll echo the json body back to the user in a 'user' key.
#     return {'user': user_as_json}
#
# See the README documentation for more examples.
#
