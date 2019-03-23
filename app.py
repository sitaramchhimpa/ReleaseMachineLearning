import numpy as np
import sklearn
import pickle
from sklearn.neural_network import MLPClassifier
from flask import Flask, request, jsonify
from numpy import float32, uint32

def predict_release_status(qa,uat,prod,count):
    output = {'release_Status_result':0,
              'message':'success'}
    x_input = np.array([qa,uat,prod,count]).reshape(1,4);

    filename = 'release_status.pkl'
    ml = pickle.load(open(filename,'rb'))

    output['release_Status_result'] = ml.predict(x_input)[0]
    print( output )
    #return ['Release_status_Result', np.int(output['release_Status_result'].item())]
    return output

app = Flask(__name__)

@app.route("/")
def index():
    return  "Release Status!!!"

@app.route("/release_status", methods=['GET'])
def find_release_status():
    body = request.get_data()
    header = request.headers

    try:
        qa = int(request.args['qa'])
        uat = int(request.args['uat'])
        prod = int(request.args['prod'])
        count = qa + uat*2 + prod*5
        res = predict_release_status(qa,uat,prod,count)
    except:
        res = {
            'success' : False,
            'message' : 'Unknown Error'
        }

    return  jsonify(res)

if __name__ == "__main__":
    app.run(debug=True, port=8080)