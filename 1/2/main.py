import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Инициализация значений
class_1_x1_mean = 24
class_1_x2_mean = 16
class_1_variance = 3

class_2_x1_mean = 11
class_2_x2_mean = 6
class_2_variance = 2

class_1_num_points = 60
class_2_num_points = 40

class_1_x1_points = np.random.normal(class_1_x1_mean, class_1_variance, class_1_num_points)
class_1_x2_points = np.random.normal(class_1_x2_mean, class_1_variance, class_1_num_points)

class_2_x1_points = np.random.normal(class_2_x1_mean, class_2_variance, class_2_num_points)
class_2_x2_points = np.random.normal(class_2_x2_mean, class_2_variance, class_2_num_points)

# Построение графика с точками 2-х классов
plt.scatter(class_1_x1_points, class_1_x2_points, color='blue', label='Точки класса -1')
plt.scatter(class_2_x1_points, class_2_x2_points, color='red', label='Точки класса 1')

plt.xlabel('Ось X1')
plt.ylabel('Ось X2')
plt.title('График с точками двух классов')
plt.legend()

plt.show()

# Создание и обучение модели
x_general = np.vstack((
    np.column_stack((class_1_x1_points, class_1_x2_points)),
    np.column_stack((class_2_x1_points, class_2_x2_points))
))
y_general = np.array([-1] * class_1_num_points + [1] * class_2_num_points)

X_train, X_test, y_train, y_test = train_test_split(x_general, y_general, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:")
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Класс -1', 'Класс 1'], yticklabels=['Класс -1', 'Класс 1'])
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.title('Матрица ошибок')
plt.show()

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Построение ROC-кривой
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (площадь = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()

# Построение PR-кривой
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
average_precision = average_precision_score(y_test, y_pred_proba, pos_label=1)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'PR кривая (средняя точность = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривая')
plt.legend(loc="lower left")
plt.show()