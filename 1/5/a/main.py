import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Загрузка данных
data = pd.read_csv('glass.csv')

# 2. Предобработка данных
# Исключаем первую колонку (предполагаем, что это индекс)
X = data.iloc[:, 1:-1]  # Все колонки, кроме первой и последней (признаки)
y = data['Type']        # Последняя колонка (целевая переменная)

# 3. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Построение и визуализация дерева решений
# Создаем модель дерева решений
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Визуализация дерева
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(i) for i in clf.classes_])
plt.title("Дерево решений")
plt.show()

# 5. Оценка точности модели
y_pred = clf.predict(X_test)
print(f'Точность модели: {accuracy_score(y_test, y_pred):.2f}')

# 6. Исследование зависимости точности от параметров
# Параметры для поиска
param_grid = {
    'criterion': ['gini', 'entropy'],  # Критерий расщепления
    'max_depth': [None, 2, 4, 6, 8, 10],  # Максимальная глубина дерева
    'min_samples_split': [2, 5, 10],  # Минимальное количество образцов для разделения
    'min_samples_leaf': [1, 2, 4]  # Минимальное количество образцов в листе
}

# Поиск по сетке
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Лучшие параметры
print(f'Лучшие параметры: {grid_search.best_params_}')

# Оценка точности с лучшими параметрами
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)
print(f'Точность с лучшими параметрами: {accuracy_score(y_test, y_pred_best):.2f}')

# 7. Визуализация дерева с лучшими параметрами (опционально)
plt.figure(figsize=(20, 10))
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=[str(i) for i in best_clf.classes_])
plt.title("Дерево решений с лучшими параметрами")
plt.show()