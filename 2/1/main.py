import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam


# Функция для загрузки и подготовки данных
def load_data(filename):
    data = pd.read_csv(filename)
    X = data[['X1', 'X2']].values
    y = data['class'].values
    # Преобразуем метки классов из {-1, 1} в {0, 1}
    y = (y + 1) // 2
    return X, y


# Функция для построения и обучения модели
def train_model(X_train, y_train, activation='sigmoid', optimizer='sgd', epochs=100):
    model = Sequential()
    model.add(Dense(1, input_dim=2, activation=activation))

    if optimizer == 'sgd':
        opt = SGD(learning_rate=0.01)
    else:
        opt = Adam(learning_rate=0.01)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.2)

    return history


# Создаем графики для одного датасета
def plot_dataset_results(X, y, dataset_name):
    # Масштабируем данные
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Разделяем на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Комбинации параметров
    activations = ['sigmoid', 'relu', 'tanh']
    optimizers = ['sgd', 'adam']
    combinations = [(a, o) for a in activations for o in optimizers]

    # Обучаем модели и сохраняем истории
    histories = {}
    for activation, optimizer in combinations:
        key = f"{activation}/{optimizer}"
        histories[key] = train_model(X_train, y_train, activation, optimizer, epochs=100)

    # Создаем графики
    plt.figure(figsize=(18, 12))

    for i, (activation, optimizer) in enumerate(combinations, 1):
        key = f"{activation}/{optimizer}"
        history = histories[key]

        plt.subplot(3, 2, i)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'{activation}/{optimizer}')
        plt.xlabel('Эпохи')
        plt.ylabel('Точность')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Загружаем данные
X0, y0 = load_data('nn_0.csv')
X1, y1 = load_data('nn_1.csv')

# Создаем графики для каждого датасета
plot_dataset_results(X0, y0, 'nn_0.csv')
plot_dataset_results(X1, y1, 'nn_1.csv')