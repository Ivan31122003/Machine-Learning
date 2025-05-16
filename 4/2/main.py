import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
data = pd.read_csv('vehicle.csv')
X = data.drop('Class', axis=1)
y = data['Class']

# Преобразование строковых меток в числовые
le = LabelEncoder()
y = le.fit_transform(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Базовые классификаторы (только те, что поддерживают sample_weight)
base_classifiers = {
    'Decision Tree': DecisionTreeClassifier(max_depth=1, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42)
}

# Диапазон количества классификаторов
n_estimators_range = [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
results = {name: [] for name in base_classifiers}

# Проведение экспериментов
for name, clf in base_classifiers.items():
    print(f"\nИсследуем базовый классификатор: {name}")
    for n in n_estimators_range:
        boosting = AdaBoostClassifier(
            estimator=clf,
            n_estimators=n,
            random_state=42
        )
        boosting.fit(X_train, y_train)
        y_pred = boosting.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name].append(acc)
        print(f"Количество классификаторов: {n:3d}, Точность: {acc:.4f}")

# Визуализация результатов
plt.figure(figsize=(12, 7))
for name, accuracies in results.items():
    plt.plot(n_estimators_range, accuracies, label=name, marker='o', linewidth=2)

plt.title('Зависимость качества AdaBoost от количества классификаторов', fontsize=14)
plt.xlabel('Количество классификаторов в ансамбле', fontsize=12)
plt.ylabel('Точность', fontsize=12)
plt.xticks(n_estimators_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('boosting_performance.png', dpi=300)
plt.show()

# Анализ оптимального количества классификаторов
print("\nАнализ оптимального количества классификаторов:")
for name, accuracies in results.items():
    best_n = n_estimators_range[np.argmax(accuracies)]
    best_acc = max(accuracies)
    print(f"{name}: оптимальное количество - {best_n} (точность {best_acc:.4f})")