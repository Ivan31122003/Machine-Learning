import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def load_data(file_path, data_type = 'str'):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            features = parts[:-1]
            label = parts[-1]
            data.append(features)
            labels.append(label)
    return np.array(data, dtype=data_type), np.array(labels)

def do_variability_datasets(x, y, rows_num):
    sets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    selection = []
    precision = []
    for set in sets:
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=set, test_size=set / 4,
                                                            random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        selection.append(set * 1.25 * rows_num)
        precision.append(accuracy)
        print(f'Точность модели: {accuracy * 100:.2f}%')
    return selection, precision

def create_plot(selection, precision):
    plt.figure(figsize=(10, 6))
    plt.plot(selection, precision, marker='o', linestyle='-', color='b')
    plt.title('Зависимость точности модели от размера обучающей выборки')
    plt.xlabel('Выборка')
    plt.ylabel('Точность модели')
    plt.grid(True)
    plt.show()