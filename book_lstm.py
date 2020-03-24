import numpy as np 
from keras.models import Sequential 
from keras.layers import LSTM 
from keras.layers import Dense





def split_sequence(sequence, n_steps): 
    X, Y = list(), list() 
    for i in range(len(sequence)):
    # find the end of this pattern 
        end_ix = i + n_steps # check if we are beyond the sequence 
        if end_ix > len(sequence)-1: 
            break # gather input and output parts of the pattern 
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] 
        X.append(seq_x) 
        Y.append(seq_y) 
    return np.array(X), np.array(Y)


n_steps = 3 
n_features = 1
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90] 

X, Y = split_sequence(raw_seq, n_steps)
X = X.reshape((X.shape[0], X.shape[1], n_features))
#print(X)
model = Sequential() 
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features))) 
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mse')
model.fit(X, Y, epochs=200, verbose=0)
x_input = np.array([70, 80, 90]) 
x_input = x_input.reshape((1, n_steps, n_features))
prediction = model.predict(x_input, verbose=0) 
print(prediction)
