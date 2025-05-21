import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Загрузка данных
data = pd.read_csv('longley.csv')

# 2. Удаление столбца Population
data = data.drop('Population', axis=1)

# 3. Разделение на признаки (X) и целевую переменную (y)
X = data.drop('Employed', axis=1)
y = data['Employed']

# 4. Разделение на обучающую и тестовую выборки (50/50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Обучение линейной регрессии
lr = LinearRegression()
lr.fit(X_train, y_train)

# Прогнозы и ошибки для линейной регрессии
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

mse_train_lr = mean_squared_error(y_train, y_train_pred_lr)
mse_test_lr = mean_squared_error(y_test, y_test_pred_lr)

# 6. Гребневая регрессия с точно заданными значениями λ
lambda_values = [10 ** (-3 + 0.2 * i) for i in range(26)]  # λ = 10^(-3 + 0.2i), i=0,...,25
train_errors = []  # Ошибки на обучающей выборке
test_errors = []  # Ошибки на тестовой выборке

for lambda_val in lambda_values:
    ridge = Ridge(alpha=lambda_val)
    ridge.fit(X_train, y_train)

    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# 7. Построение графиков
plt.figure(figsize=(14, 6))

# График ошибок гребневой регрессии
plt.subplot(1, 2, 1)
plt.semilogx(lambda_values, train_errors, label='Ошибка обучения', marker='o', markersize=4)
plt.semilogx(lambda_values, test_errors, label='Ошибка теста', marker='o', markersize=4)
plt.xlabel('λ (лог. шкала)')
plt.ylabel('Среднеквадратичная ошибка (MSE)')
plt.title('Зависимость ошибок от параметра λ\nГребневая регрессия')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.legend()

# 8. Нахождение оптимального λ
optimal_idx = np.argmin(test_errors)
optimal_lambda = lambda_values[optimal_idx]
print(f"Оптимальное значение λ: {optimal_lambda:.6f}")
print(f"Соответствующая тестовая ошибка: {test_errors[optimal_idx]:.4f}")

# 9. Вывод результатов
print("\nЛинейная регрессия:")
print(f"Ошибка обучения: {mse_train_lr:.4f}")
print(f"Ошибка теста: {mse_test_lr:.4f}\n")

print("Гребневая регрессия (крайние значения):")
print(f"λ_min = {lambda_values[0]:.6f}: Ошибка обучения = {train_errors[0]:.4f}, Ошибка теста = {test_errors[0]:.4f}")
print(
    f"λ_opt = {optimal_lambda:.6f}: Ошибка обучения = {train_errors[optimal_idx]:.4f}, Ошибка теста = {test_errors[optimal_idx]:.4f}")
print(
    f"λ_max = {lambda_values[-1]:.6f}: Ошибка обучения = {train_errors[-1]:.4f}, Ошибка теста = {test_errors[-1]:.4f}")

plt.tight_layout()
plt.show()