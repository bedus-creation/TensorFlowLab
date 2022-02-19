from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
from matplotlib import pyplot
from datetime import datetime
%matplotlib notebook

def slide_window(data, lag:int = 1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df = df.drop(0)
    return df

def prepare_model():
    model = Sequential()
    # The next line is optional that we are adding more hidden layers in our model.
#     model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(3, 1)))
    model.add(LSTM(23, activation='relu', input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model

df = pd.DataFrame([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], columns = ['temp'])

data = slide_window(df, 3)

# Eliminating the null values
X = data.values[3:-1:,0:3]
Y = data.values[3:-1:, -1]


n_features = 1 # As univariate data model

X = X.reshape((X.shape[0], X.shape[1], n_features))

model = prepare_model()
model.fit(X, Y, epochs=200, verbose=2)

# Prediction
x_input = pd.DataFrame([70, 80, 90])
x_input = x_input.values.reshape((1, 3, n_features))

yhat = model.predict(x_input, verbose=2)

print(yhat)
