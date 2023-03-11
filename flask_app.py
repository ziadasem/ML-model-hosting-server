
from flask import Flask, jsonify, request
import json
import pickle
import numpy as np



import pandas as pd


refrence_column_old = ['result',
                   'lat-position', 'lng-position', #'index',
        'c8:0c:c8:13:74:cc',
       'b0:ac:d2:42:96:0c', '58:ba:d4:b4:33:5c', '38:54:9b:32:33:f0',
       'a4:f3:3b:bc:6d:18', '38:54:9b:32:33:f1', '20:e8:82:ba:b6:6a',
       '08:10:76:46:db:ed', 'c0:fd:84:d5:3d:17', '8c:15:c7:05:dd:bc',
       '58:d0:61:8f:7b:d0', '1c:a5:32:78:b2:35', 'b0:ac:d2:73:f1:13',
       '2c:78:0e:d4:d4:be', '58:d0:61:8f:7b:cc', '2a:77:77:9b:74:84',
       '64:6d:6c:0e:08:a0', '28:77:77:cb:87:c8', 'b4:f5:8e:7f:22:88',
       'c8:0c:c8:3d:ce:14', 'd4:6b:a6:c8:8c:68']

refrence_column = [ 'lat-position', 'lng-position', 'c8:0c:c8:13:74:cc',
       'b0:ac:d2:42:96:0c', '58:ba:d4:b4:33:5c', '38:54:9b:32:33:f0',
       'a4:f3:3b:bc:6d:18', '38:54:9b:32:33:f1', '20:e8:82:ba:b6:6a',
       '08:10:76:46:db:ed', 'c0:fd:84:d5:3d:17', '8c:15:c7:05:dd:bc',
       '58:d0:61:8f:7b:d0', '1c:a5:32:78:b2:35', 'b0:ac:d2:73:f1:13',
       '2c:78:0e:d4:d4:be', '58:d0:61:8f:7b:cc', '2a:77:77:9b:74:84',
       '64:6d:6c:0e:08:a0', '28:77:77:cb:87:c8', 'b4:f5:8e:7f:22:88',
       'c8:0c:c8:3d:ce:14', 'd4:6b:a6:c8:8c:68']

labels = [ ["KNN0:RF0", "KNN0:RF1", "KNN0:RF2", "KNN0:RF3"], ["KNN1:RF0", "KNN1:RF1", "KNN1:RF2", "KNN1:RF3"],
            ["KNN2:RF0", "KNN2:RF1", "KNN2:RF2", "KNN2:RF3"], ["KNN3:RF0", "KNN3:RF1", "KNN3:RF2", "KNN3:RF3"]
           ]

def json_to_dataframe(json_obj, columns):
    # Convert the JSON object to a Pandas Series
    series = pd.Series(json_obj)

    # Convert the Series to a DataFrame with one row
    df = pd.DataFrame(series).T

    # Add any columns specified in the "columns" list that are not already in the DataFrame
    for column in columns:
        if column not in df.columns:
            df[column] = -95

    # Remove brackets from columns that contain lists
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

    # Order the DataFrame based on the specified columns
    df = df[columns]
    # df=sc.fit_transform(df)

    return df

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

            input_df = json_to_dataframe(request_data, refrence_column)

            #input_df_list = [30.0418759, 31.3741326,	-55,	-70,	-70,	-73,	-95,	-88,	-71,	-95,
            #           -95,	-88,	-95,	-74,	-95,	-95,	-95,	-95,	-95,	-95,
            #          -95,	-87,	-95]
            #input_df = pd.DataFrame(input_df_list)
            #input_df = np.array(input_df)
            #input_df = input_df.transpose()

            rf_model_index = knn_model.predict(input_df)[0]
            current_rf_model =RF_Models[rf_model_index]
            input_df.insert(2, "index", [rf_model_index])
            rf_result = current_rf_model.predict(input_df)[0]

            return jsonify({"label": labels[rf_model_index][rf_result]})
        except Exception as e:
            return str(e)
    else:
        text = input
        print(input)
        return jsonify({'output' : text})

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True, port=80)
