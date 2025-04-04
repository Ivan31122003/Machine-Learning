import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Функция для загрузки данных
def load_data(filename):
    data = []
    labels = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            features = parts[1:-1]
            label = parts[-1]
            data.append(features)
            labels.append(label)
    return np.array(data, dtype='float'), np.array(labels)

# Функция для визуализации разделяющей поверхности
def plot_decision_boundary(X, y, model, title):
    h = .02  # Шаг сетки
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# Функция для отображения матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.show()

# Загрузка данных
X_train, y_train = load_data('svmdata_d.txt')
X_test, y_test = load_data('svmdata_d_test.txt')

# Преобразование текстовых меток в числовые
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Определение ядер и их параметров
kernels = [
    ('poly (degree=1)', {'kernel': 'poly', 'degree': 1}),
    ('poly (degree=2)', {'kernel': 'poly', 'degree': 2}),
    ('poly (degree=3)', {'kernel': 'poly', 'degree': 3}),
    ('poly (degree=4)', {'kernel': 'poly', 'degree': 4}),
    ('poly (degree=5)', {'kernel': 'poly', 'degree': 5}),
    ('sigmoid', {'kernel': 'sigmoid'}),
    ('RBF', {'kernel': 'rbf'})
]

# Обучение и визуализация для каждого ядра
for kernel_name, params in kernels:
    clf = SVC(**params)
    clf.fit(X_train, y_train)

    # Визуализация разделяющей поверхности
    plot_decision_boundary(X_train, y_train, clf, f'SVM с ядром: {kernel_name}')

    # Предсказания и оценка точности
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Ядро: {kernel_name}")
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")

    # Визуализация матрицы ошибок для обучающей выборки
    plot_confusion_matrix(y_train, y_train_pred, f'Матрица ошибок (обучение): {kernel_name}')

    # Визуализация матрицы ошибок для тестовой выборки
    plot_confusion_matrix(y_test, y_test_pred, f'Матрица ошибок (тест): {kernel_name}')

    print("-" * 50)