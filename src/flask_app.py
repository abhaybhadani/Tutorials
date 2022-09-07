from flask import Flask
import numpy as np
from keras.models import model_from_json
from flask import jsonify, request
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import string
import pickle

# import requests

app = Flask(__name__)

# load json and create model

# Load the model architecture    
json_file = open('./models/test_model.json', 'r')
loaded_model = model_from_json(json_file.read())
json_file.close()

# load weights into new model
loaded_model.load_weights("./models/test_model.h5")
print("Loaded model from disk")

# loading tokens
with open('./models/test_model.token', 'rb') as handle:
    tokenizer = pickle.load(handle)

print("Loaded model from disk")

sentiment_labels = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
sentiment_labels_encoding = [0, 1, 2, 3, 4, 5]


def preprocessingText1(sentences):
    sentences = sentences.apply(lambda sequence:
                                              [ltrs.lower() for ltrs in sequence if ltrs not in string.punctuation])
    sentences = sentences.apply(lambda wrd: ''.join(wrd))
    return sentences


@app.route("/predict_sentiment", methods=['POST'])
def predict_sentiment():
    try:
        params = json.loads(request.get_data())
        text = params.get("query")
        print('Query: ',text)
    except Exception as e:
        print('text= ', text)
        text="I love text analysis"

    tw = preprocessingText1(pd.Series(text))
    tw = tokenizer.texts_to_sequences(tw)
    tw = pad_sequences(tw, maxlen= 63)
    prob =loaded_model.predict(tw)
    idx=pd.Series(prob[0]).idxmax()
    #print(tw, prob, idx, sentiment_labels[idx])
    prob_score= str(max(prob[0]))
    return json.dumps({'Label': sentiment_labels[idx], 'model probability score': prob_score})
    #return jsonify({'Label': sentiment_labels[idx], 'model probability score':prob})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

test_sentence1 = "I enjoyed my journey on this flight."
print(predict_sentiment(test_sentence1))
