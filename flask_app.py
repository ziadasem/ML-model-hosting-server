
from flask import Flask, jsonify, request
import json
import numpy as np
import os
import pickle

app = Flask(__name__)

def deserialize(file_path):
    # de-serialize pkl file into an object called model using pickle
    with open(file_path, 'rb') as handle:
      model = pickle.load(handle)
    return model


RF_Models = [
    deserialize('./model_pickle_RF0.obj'),
    deserialize('./model_pickle_RF1.obj'),
    deserialize('./model_pickle_RF2.obj'),
    deserialize('./model_pickle_RF3.obj'),
    deserialize('./model_pickle_RF4.obj')
]

@app.route('/')
def hello_world():
    return 'Hi'

@app.route('/', methods = ['GET', 'POST'])
def nameRoute():
    global input

    if(request.method == 'POST'):
        try:
            knn_model = deserialize('./model_pickle_KNN.obj')
            request_data = request.data

            request_data = json.loads(request_data)
            knn_input = request_data['input_knn']
            input = request_data['input']
            input = np.array(input)
            rf_model_index = knn_model.predict([knn_input])[0]
            current_rf_model =RF_Models[rf_model_index]
            result = current_rf_model.predict([input])
            return jsonify({'output': str(result[0])})
        except Exception as e:
            return str(e)
    else:
        text = input
        print(input)
        return jsonify({'output' : text})

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True, port=80)
