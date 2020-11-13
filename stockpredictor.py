import numpy as np
import pandas as pd 
import matplotlib.pyplot as mpl 
from sklearn.linear_model import LinearRegression, SGDClassifier

x_read_data = pd.read_csv('stocks/AAPL.csv', usecols=['Open','High','Low'])
y_read_data = pd.read_csv('stocks/AAPL.csv', usecols=['Close'])
x_data = x_read_data.values 
y_data = y_read_data.values 

X_train = x_data[:]
Y_train = y_data[:]

X_test = np.array([[257.26, 278.41, 256.37]])

reg = LinearRegression().fit(X_train, Y_train)

print(reg.score(X_train, Y_train))
print(reg.predict(X_test))