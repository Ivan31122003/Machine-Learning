import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 1. Загрузка данных
train_data = pd.read_csv('bank_scoring_train.csv', sep='\t')
test_data = pd.read_csv('bank_scoring_test.csv', sep='\t')

# 2. Предобработка данных
# Заполнение пропущенных значений медианой
imputer = SimpleImputer(strategy='median')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Разделение на признаки и целевую переменную
X_train = train_data.drop('SeriousDlqin2yrs', axis=1)
y_train = train_data['SeriousDlqin2yrs']
X_test = test_data.drop('SeriousDlqin2yrs', axis=1)
y_test = test_data['SeriousDlqin2yrs']

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Функция для визуализации матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Да', 'Нет'],
                yticklabels=['Да', 'Нет'])
    plt.title(f'Матрица ошибок - {model_name}')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Истинные значения')
    plt.show()


# 4. Оценка качества моделей
def evaluate_model(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\nРезультаты модели {model_name}:")
    print("=" * 50)
    print("Отчет о классификации:")
    print(classification_report(y_test, y_pred))

    # Визуализация матрицы ошибок
    plot_confusion_matrix(y_test, y_pred, model_name)

    print(f"\nF1-мера: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("=" * 50)


# 3. Построение и обучение моделей
# Наивный Байесовский классификатор
print("Обучение наивного Байесовского классификатора...")
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
evaluate_model(nb_clf, "Наивный Байес", X_test, y_test)

# Метод k-ближайших соседей
print("\nОбучение метода k-ближайших соседей...")
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
evaluate_model(knn_clf, "Метод k-NN", X_test, y_test)

# Сравнение точности моделей
print("\nСравнение точности моделей:")
print(f"Наивный Байес: {nb_clf.score(X_test, y_test):.4f}")
print(f"Метод k-NN: {knn_clf.score(X_test, y_test):.4f}")
