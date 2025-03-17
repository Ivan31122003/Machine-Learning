import numpy as np
from sklearn.preprocessing import LabelEncoder
from functions import load_data
from functions import do_variability_datasets
from functions import create_plot

def encode_features(x):
    x_encoded = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] == 'x':
                x_encoded[i, j] = 1
            elif x[i, j] == 'o':
                x_encoded[i, j] = -1
            else:
                x_encoded[i, j] = 0
    return x_encoded

file_path = 'C:/Users/Windows/Desktop/1 лаба/tic_tac_toe.txt'
x, y = load_data(file_path)

x_encoded = encode_features(x)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

selection, precision = do_variability_datasets(x_encoded, y_encoded, 958)

create_plot(selection, precision)