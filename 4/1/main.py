import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
data = pd.read_csv('glass.csv')
X = data.drop(['Id', 'Type'], axis=1)
y = data['Type']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Определение базовых классификаторов
base_classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'k-NN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42)
}

# Диапазон количества классификаторов
n_estimators_range = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70]
results = {name: [] for name in base_classifiers}

# Проведение экспериментов
for name, clf in base_classifiers.items():
    print(f"Исследуем базовый классификатор: {name}")
    for n in n_estimators_range:
        bagging = BaggingClassifier(
            estimator=clf,
            n_estimators=n,
            random_state=42
        )
        bagging.fit(X_train, y_train)
        y_pred = bagging.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name].append(acc)
        print(f"Количество классификаторов: {n}, Точность: {acc:.4f}")

# Визуализация результатов
plt.figure(figsize=(12, 7))
for name, accuracies in results.items():
    plt.plot(n_estimators_range, accuracies, label=name, marker='o', linewidth=2)

plt.title('Зависимость качества классификации от количества классификаторов в ансамбле', fontsize=14)
plt.xlabel('Количество классификаторов в ансамбле', fontsize=12)
plt.ylabel('Точность', fontsize=12)
plt.xticks(n_estimators_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Сохранение графика
plt.savefig('bagging_performance.png', dpi=300)
plt.show()

# Анализ оптимального количества классификаторов
print("\nАнализ оптимального количества классификаторов:")
for name, accuracies in results.items():
    best_n = n_estimators_range[np.argmax(accuracies)]
    best_acc = max(accuracies)
    print(f"{name}: оптимальное количество - {best_n} (точность {best_acc:.4f})")

# Дополнительный анализ для Decision Tree (наиболее перспективного)
print("\nДетальный анализ для Decision Tree:")
dt_acc = results['Decision Tree']
improvement = [dt_acc[i] - dt_acc[0] for i in range(len(dt_acc))]
for i, n in enumerate(n_estimators_range):
    print(f"n={n}: Улучшение относительно 1 классификатора: {improvement[i]:.4f}")

# Поиск точки, где улучшение становится менее 0.01
threshold = 0.01
for i in range(1, len(improvement)):
    if improvement[i] - improvement[i-1] < threshold:
        print(f"\nДля Decision Tree после {n_estimators_range[i-1]} классификаторов улучшение становится менее {threshold}")
        break