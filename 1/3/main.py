import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def create_plot(x, y, title, x_lable, y_label):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.xticks(x)
    plt.yticks(y)
    plt.title(title)
    plt.xlabel(x_lable)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

data = pd.read_csv('glass.csv')
print(data)

first_column_name = data.columns[0]
data = data.drop(columns=[first_column_name])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

neighbors = []
precision = []

metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski']

# Исследование влияния метрики расстояния на точность
for metric in metrics:
    print(f"\nИсследование метрики: {metric}")
    neighbors = []
    precision = []

    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i, metric=metric)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Точность модели (k={i}): {accuracy * 100:.2f}%')
        neighbors.append(i)
        precision.append(accuracy)

    create_plot(neighbors, precision, f'Зависимость точности от кол-ва соседей (метрика: {metric})', 'Кол-во соседей', 'Точность')

create_plot(neighbors, precision, 'Зависимость точности классификации от кол-ва ближайших соседей', 'Кол-во соседей', 'Вероятность ошибки')

knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train, y_train)

x_set = [1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]

x_set_2d = np.array(x_set).reshape(1, -1)

y_pred_one_row = knn1.predict(x_set_2d)
print(y_pred_one_row)
