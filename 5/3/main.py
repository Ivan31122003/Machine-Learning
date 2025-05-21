import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Загрузка данных
try:
    data = pd.read_csv('cygage.txt', delim_whitespace=True)
except:
    print("Ошибка загрузки файла. Убедитесь, что файл 'cygage.txt' находится в правильной директории.")
    exit()

# Проверка наличия нужных столбцов
required_columns = ['calAge', 'Depth', 'Weight']
if not all(col in data.columns for col in required_columns):
    print("Файл должен содержать столбцы: 'calAge', 'Depth', 'Weight'")
    exit()

# Подготовка данных
X = data[['Depth']].values
y = data['calAge'].values
weights = data['Weight'].values

# Модель без учета весов
model_unweighted = LinearRegression()
model_unweighted.fit(X, y)
y_pred_unweighted = model_unweighted.predict(X)

# Модель с учетом весов
model_weighted = LinearRegression()
model_weighted.fit(X, y, sample_weight=weights)
y_pred_weighted = model_weighted.predict(X)

# Оценка качества моделей
r2_unweighted = r2_score(y, y_pred_unweighted)
r2_weighted = r2_score(y, y_pred_weighted)
mse_unweighted = mean_squared_error(y, y_pred_unweighted)
mse_weighted = mean_squared_error(y, y_pred_weighted, sample_weight=weights)

# Вывод результатов
print("Результаты регрессии:")
print(f"Модель без весов: R² = {r2_unweighted:.4f}, MSE = {mse_unweighted:.2f}")
print(f"Модель с весами: R² = {r2_weighted:.4f}, MSE (взвешенное) = {mse_weighted:.2f}")
print("\nКоэффициенты:")
print(f"Без весов: intercept = {model_unweighted.intercept_:.2f}, slope = {model_unweighted.coef_[0]:.4f}")
print(f"С весами: intercept = {model_weighted.intercept_:.2f}, slope = {model_weighted.coef_[0]:.4f}")

# Визуализация
plt.figure(figsize=(12, 6))
plt.scatter(X, y, c='blue', label='Данные', alpha=0.7)
plt.plot(X, y_pred_unweighted, 'r-', label='Без весов')
plt.plot(X, y_pred_weighted, 'g--', label='С весами')
plt.xlabel('Глубина залегания')
plt.ylabel('Возраст отложений')
plt.title('Сравнение моделей')
plt.legend()
plt.grid(True)
plt.show()

# Дополнительная диагностика
residuals_unweighted = y - y_pred_unweighted
residuals_weighted = y - y_pred_weighted

plt.figure(figsize=(12, 6))
plt.scatter(y_pred_unweighted, residuals_unweighted, c='red', label='Без весов', alpha=0.5)
plt.scatter(y_pred_weighted, residuals_weighted, c='green', label='С весами', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Диаграмма остатков')
plt.legend()
plt.grid(True)
plt.show()