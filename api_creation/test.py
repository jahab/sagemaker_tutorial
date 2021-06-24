import requests
import json
# Define test JSON
vid_name='Final_1280_720.mp4' ####uploaded video name in S3 bucket
input_video = {'input':{'video':vid_name}}

input_json = json.dumps(input_video)
# Define your api URL here

api_url = 'https://7w4hrf8v35.execute-api.us-east-1.amazonaws.com/api/'
res = requests.post(api_url, json=input_json)
output_api = res.text
# print(output_api)

