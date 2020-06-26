import numpy as np

from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app_route('/predict', methods=['POST'])
def predict():
    ''' 
    for rendering results in HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    return render_template('index.html', prediction_text= 'Student progress score ${}'.format(prediction))
    
    
if __name__ == "__main_":
    app.run(debug=True)
