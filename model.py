import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


dataset = pd.read_csv('/Users/allan/Desktop/AI-ML-DL/Flask-Heroku-Model/StudentData.csv')
dataset.head()  

X = dataset.iloc[:, :3]


def convert_to_int(x):
    word_dict = {'good':1, 'medium':2, 'low':3}
    return word_dict[x]

dataset['outcome'] = dataset['outcome'].apply(lambda x: convert_to_int(x))

y = dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor, open('model.pkl', 'wb')) #wb means write byte

#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,100,100]]))

