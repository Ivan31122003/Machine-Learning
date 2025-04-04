import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

def plot_decision_boundary(X, y, model, title):
    h = .02
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

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(title)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.show()

# Загрузка данных
X_train, y_train = load_data('svmdata_a.txt')
X_test, y_test = load_data('svmdata_a_test.txt')

# Преобразование текстовых меток в числовые
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Построение модели SVM с линейным ядром
clf_linear = SVC(kernel='linear')
clf_linear.fit(X_train, y_train)

# Визуализация разделяющей поверхности
plot_decision_boundary(X_train, y_train, clf_linear, 'SVM с линейным ядром')

# Количество опорных векторов
print(f"Количество опорных векторов: {clf_linear.support_vectors_.shape[0]}")

# Предсказания и визуализация матрицы ошибок
y_train_pred = clf_linear.predict(X_train)
y_test_pred = clf_linear.predict(X_test)

plot_confusion_matrix(y_train, y_train_pred, 'Матрица ошибок (обучающая выборка)')
plot_confusion_matrix(y_test, y_test_pred, 'Матрица ошибок (тестовая выборка)')