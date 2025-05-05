import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

# Включим eager execution для TensorFlow
tf.config.run_functions_eagerly(True)

# Загрузка данных
try:
    data0 = pd.read_csv('nn_0.csv')
    X0 = data0[['X1', 'X2']].values.astype('float32')
    y0 = data0['class'].values.astype('float32')

    data1 = pd.read_csv('nn_1.csv')
    X1 = data1[['X1', 'X2']].values.astype('float32')
    y1 = data1['class'].values.astype('float32')
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    exit()


# Функция для отображения исходных данных
def plot_initial_data(X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Класс 1', alpha=0.6)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Класс -1', alpha=0.6)
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Покажем исходные данные
print("\nВизуализация исходных данных:")
plot_initial_data(X0, y0, 'nn_0.csv')
plot_initial_data(X1, y1, 'nn_1.csv')


# Функция для разделения данных
def simple_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]


# Функция для создания модели
def create_model(activation='sigmoid', optimizer_type='adam'):
    model = Sequential([Dense(1, input_shape=(2,), activation=activation)])

    if optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=0.01)
    else:
        optimizer = Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Функция для обучения и визуализации
def train_and_visualize(X, y, dataset_name):
    activations = ['sigmoid', 'relu', 'tanh']
    optimizers = ['sgd', 'adam']

    plt.figure(figsize=(15, 10))
    results = []

    for i, activation in enumerate(activations, 1):
        for j, optimizer in enumerate(optimizers, 1):
            try:
                # Создаем и обучаем модель
                model = create_model(activation, optimizer)
                X_train, X_test, y_train, y_test = simple_split(X, y)
                history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)

                # Оценка модели
                _, test_acc = model.evaluate(X_test, y_test, verbose=0)
                results.append((activation, optimizer, test_acc))

                # График обучения
                plt.subplot(3, 2, (i - 1) * 2 + j)
                plt.plot(history.history['accuracy'], label='Train')
                plt.plot(history.history['val_accuracy'], label='Validation')
                plt.title(f'{activation}/{optimizer}\nТочность: {test_acc:.2f}')
                plt.xlabel('Эпохи')
                plt.ylabel('Точность')
                plt.legend()
                plt.grid()

            except Exception as e:
                print(f"Ошибка для {activation}/{optimizer}: {str(e)}")
                results.append((activation, optimizer, 0.0))

    plt.tight_layout()
    plt.suptitle(f'Результаты обучения для {dataset_name}', y=1.02)
    plt.show()

    # Вывод лучшей комбинации
    best = max(results, key=lambda x: x[2])
    print(f"\nЛучшая комбинация для {dataset_name}: {best[0]}/{best[1]} с точностью {best[2]:.4f}")

    # Визуализация классификации для лучшей модели
    best_model = create_model(best[0], best[1])
    best_model.fit(X, y, epochs=100, verbose=0)
    plot_decision_boundary(best_model, X, y, f'Классификация ({best[0]}/{best[1]}) для {dataset_name}')


# Функция для отображения границы решения
def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(8, 6))

    # Создаем сетку для визуализации
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Предсказание для сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    # Отображение
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Класс 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Класс -1')
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Основная программа
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # Обучаем на первом датасете
    print("\n" + "=" * 60)
    print("Обучение на nn_0.csv")
    print("=" * 60)
    train_and_visualize(X0, y0, 'nn_0.csv')

    # Обучаем на втором датасете
    print("\n" + "=" * 60)
    print("Обучение на nn_1.csv")
    print("=" * 60)
    train_and_visualize(X1, y1, 'nn_1.csv')