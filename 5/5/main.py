import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Загрузка данных
data = pd.read_csv('eustock.csv')

# 1. Визуализация всех котировок на одном графике
plt.figure(figsize=(12, 6))
for column in data.columns:
    plt.plot(data[column], label=column)
plt.title('Ежедневные котировки фондовых бирж')
plt.xlabel('Дни')
plt.ylabel('Котировки')
plt.legend()
plt.grid()
plt.show()

# Подготовка данных для регрессии
X = np.arange(len(data)).reshape(-1, 1)  # Временная ось (дни)

# 2. Линейная регрессия для каждой биржи отдельно с визуализацией
models = {}
plt.figure(figsize=(12, 6))
for column in data.columns:
    model = LinearRegression()
    model.fit(X, data[column])
    models[column] = {
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r_squared': model.score(X, data[column])
    }

    # Визуализация на отдельных графиках
    plt.figure(figsize=(12, 4))
    plt.scatter(X, data[column], label='Данные', s=10)
    plt.plot(X, model.predict(X), color='red', label='Линейная регрессия')
    plt.title(f'Линейная регрессия для {column}')
    plt.xlabel('Дни')
    plt.ylabel('Котировки')
    plt.legend()
    plt.grid()
    plt.show()

# 3. Визуализация всех регрессий на одном графике
plt.figure(figsize=(12, 6))
colors = ['blue', 'orange', 'green', 'red']
i = 0
for column in data.columns:
    model = models[column]['model']
    plt.plot(data[column], label=f'{column} (данные)', color=f'{colors[i]}', alpha=0.7)
    plt.plot(X, model.predict(X), label=f'{column} (тренд)', color=f'{colors[i]}', linestyle='--')
    i = i + 1
plt.title('Линейные регрессии для всех бирж')
plt.xlabel('Дни')
plt.ylabel('Котировки')
plt.legend()
plt.grid()
plt.show()

# 4. Многомерная регрессия (все биржи вместе)
multi_model = LinearRegression()
multi_model.fit(X, data)
multi_r_squared = multi_model.score(X, data)

# Вывод результатов
print("Результаты линейной регрессии для каждой биржи:")
for stock, result in models.items():
    print(f"{stock}:")
    print(f"  Коэффициент наклона (динамика): {result['slope']:.4f}")
    print(f"  R²: {result['r_squared']:.4f}")
    print()

print(f"Многомерная регрессия для всех бирж вместе: R² = {multi_r_squared:.4f}")

# 5. Определение биржи с наибольшей динамикой
most_dynamic = max(models.items(), key=lambda x: abs(x[1]['slope']))
print(f"\nБиржа с наибольшей динамикой: {most_dynamic[0]} (коэффициент наклона: {most_dynamic[1]['slope']:.4f})")