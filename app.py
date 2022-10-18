from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import random
import json
import numpy as np
import pickle
import requests
import json
import os
import re

classes = pickle.load(open('labels.pkl', 'rb'))

# get value from enviroment variable
tenorflow_url = os.environ.get(
    'TENSORFLOW_URL', 'http://localhost:8501/v1/models/multilable_model:predict')

predict_threshold = os.environ.get(
    'pred_threshold', "0.2")

predict_threshold = float(predict_threshold)
# Get responce from tensorflow model server


def get_responce_from_model_server(msg):
    data = json.dumps(
        {"signature_name": "serving_default", "instances": [msg]})
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        tenorflow_url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions

# create a dictionory of predection and class name


def get_prediction_dict(predictions):
    prediction_dict = {}
    for i, p in enumerate(predictions[0]):
        prediction_dict[classes[i]] = p
    return prediction_dict

# Filter the dictionary to get only the intents that are above the threshold


def filter_predictions(predictions, threshold):
    filtered_predictions = {}
    for key, value in predictions.items():
        if value > threshold:
            filtered_predictions[key] = value
    return filtered_predictions

# Convert dictionary keys to text seprated by comma


def get_text_from_dict(dict):
    text = "Predected Genres are "
    for key in dict:
        text += key + ", "
    return text

# function to clean the word of any punctuation or special characters and lowwer it


def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.lower()
    return cleaned


def chatbot_response(msg):
    msg = cleanPunc(msg)
    pred = get_responce_from_model_server(msg)
    pred = get_prediction_dict(pred)
    pred = filter_predictions(pred, predict_threshold)
    pred = get_text_from_dict(pred)
    return pred


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    run_with_ngrok(app)
    app.run()
