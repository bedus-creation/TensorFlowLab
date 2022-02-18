from keras.models import Sequential
from keras.layers import LSTM, Dense
from numpy import array

def slide_window(series, slide: int = 1):
    inputs = []
    outputs = []
    for i in range(len(series)):
        if(i+slide <len(series)):
            y = series[i+slide]
            x = series[i:i+slide]  
            
            inputs.append(x)
            outputs.append(y)
            
    return array(inputs), array(outputs),

def model():
    model = Sequential()
    model.add(LSTM(23, activation='relu', input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model
    
# Data preparation
data = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) # numpy array

# Preparation of time series data: i.e. repeating the data for 3 slide windlows gives:
# 10, 20, 30 => 30 
# 20, 30, 40 => 40
X, Y = slide_window(data, 3)

# Preperation of 3D data to fit into models.
n_features = 1 # As univariate data model

X = X.reshape((X.shape[0], X.shape[1], n_features))


# init model
model = model()
model.fit(X, Y, epochs=200, verbose=0)

# Prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, 3, n_features))
yhat = model.predict(x_input, verbose=0)

print(yhat)
