import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = np.array([float(x) for x in request.form.values()]).reshape(1,-1)
    preds = model.predict(features)
    output = round(preds[0], 2)

    return render_template('index.html', prediction_text='The number of rings is approximately {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)