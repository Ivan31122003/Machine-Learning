import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Загрузка данных
data = np.loadtxt('svmdata6.txt', skiprows=1)
X = data[:, 1].reshape(-1, 1)  # Вторая колонка как признаки
y = data[:, 2]                 # Третья колонка как целевая переменная

# Диапазон значений epsilon для исследования
epsilon_values = np.linspace(0.01, 1.0, 50)
mse_values = []

# Исследование влияния epsilon на MSE
for eps in epsilon_values:
    model = SVR(kernel='rbf', C=1, epsilon=eps)
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mse_values.append(mse)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, mse_values, marker='o', linestyle='-', color='b')
plt.title('Зависимость MSE от параметра ε (SVR с RBF ядром, C=1)')
plt.xlabel('Значение ε')
plt.ylabel('Среднеквадратичная ошибка (MSE)')
plt.grid(True)
plt.show()

# Вывод комментария
print("Анализ графика показывает, что с увеличением параметра ε:")
print("- MSE на обучающей выборке сначала постепенно увеличивается")
print("- Это соответствует ожидаемому поведению, так как большие значения ε")
print("  позволяют модели допускать большие ошибки на обучающих данных")
print("- При малых ε модель стремится точно соответствовать обучающим данным")
print("- Оптимальное значение ε следует выбирать на основе кросс-валидации")