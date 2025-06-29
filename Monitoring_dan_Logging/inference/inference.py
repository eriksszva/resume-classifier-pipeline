import requests
import json

# load serving input from serving_input_example.json
# this file should contain the input data in the format expected by the classifier API
with open("serving_input_example.json", "r") as f:
    payload = json.load(f)

# convert MLflow-style "dataframe_split" to columns + data
columns = payload["dataframe_split"]["columns"]
data = payload["dataframe_split"]["data"]

# create request format expected by FastAPI classifier
request_payload = {
    "columns": columns,
    "data": data
}

# send POST request to classifier API
response = requests.post("http://localhost:8000/predict", json=request_payload) # use port in the exporter not the classifier

# print response
if response.ok:
    print("Predictions:", response.json()["predictions"])
else:
    print("Error:", response.status_code, response.text)