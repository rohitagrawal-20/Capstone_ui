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
from transformers import BertTokenizer

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
        {"signature_name": "serving_default", "instances": msg})
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        tenorflow_url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    if predictions>=0.5:
        return "Negative"
    else:
        return "Positive"

def bert_encode(data,tokenizer,maximum_length) :
    input_ids = []
    attention_masks = []
  

    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
        
        data[i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,
        
        return_attention_mask=True,
        
      )
      
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return {
        'input_ids':input_ids,
        'attention_mask':attention_masks
    }

def importing_tokenizer():
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  return tokenizer


# function to clean the word of any punctuation or special characters and lowwer it


def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.lower()
    return [cleaned]


def chatbot_response(msg):
    msg = cleanPunc(msg)
    dic = bert_encode(msg,tokenizer,16)
    pred = get_responce_from_model_server(msg)
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
    tokenizer = importing_tokenizer()
    run_with_ngrok(app)
    app.run()
