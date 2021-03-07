import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

X = pd.read_csv('data/abalone.csv')
y = X.pop("Rings")

def encoder(gender):
    dict = {'M':0,'F':1,'I':2}
    return dict[gender]

X['Sex'] = X['Sex'].apply(lambda x: encoder(x))

model = LinearRegression()
model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict(X))


