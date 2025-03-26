import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Загрузка данных
data = pd.read_csv('spam7.csv')

# 2. Предобработка данных
X = data.drop('yesno', axis=1)  # Все колонки, кроме последней
y = data['yesno']  # Последняя колонка (целевая переменная)

# 3. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Поиск оптимальных параметров
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f'Лучшие параметры: {best_params}')
best_clf = grid_search.best_estimator_

# 5. Визуализация дерева
plt.figure(figsize=(20, 10))
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=best_clf.classes_)
plt.title("Дерево решений с лучшими параметрами")
plt.show()

# 6. Важность признаков
feature_importances = best_clf.feature_importances_
for feature, importance in zip(X.columns, feature_importances):
    print(f'{feature}: {importance:.4f}')

# 7. Оценка качества классификации
y_pred = best_clf.predict(X_test)
print(f'Точность модели: {accuracy_score(y_test, y_pred):.2f}')
print("Отчет о классификации:")
print(classification_report(y_test, y_pred))
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))