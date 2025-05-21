import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Загрузка и подготовка данных
data = pd.read_csv('JohnsonJohnson.csv')
data['year'] = data['index'].str.split(' Q').str[0].astype(float)
data['quarter'] = data['index'].str.split(' Q').str[1].astype(int)
data = data.sort_values(['year', 'quarter'])
data['time'] = data['year'] + (data['quarter'] - 1) / 4  # Создаем столбец time

# 1. График с 4 квартальными кривыми на одном изображении
plt.figure(figsize=(14, 8))
colors = ['blue', 'orange', 'green', 'red']

for q in range(1, 5):
    quarter_data = data[data['quarter'] == q]
    plt.plot(quarter_data['year'], quarter_data['value'],
             color=colors[q - 1], linewidth=2, label=f'Q{q}')

plt.title('Квартальная прибыль Johnson & Johnson (1960-1980)', fontsize=16)
plt.xlabel('Год', fontsize=14)
plt.ylabel('Прибыль', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, title='Кварталы', title_fontsize=12)
plt.xticks(np.arange(1960, 1981, 5))
plt.tight_layout()
plt.show()

# 2. 4 отдельных графика с регрессиями
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Квартальная прибыль Johnson & Johnson (1960-1980) по кварталам', fontsize=16)

quarter_models = {}
for q, ax in enumerate(axes.flat, start=1):
    quarter_data = data[data['quarter'] == q]
    X = quarter_data[['year']].values
    y = quarter_data['value'].values

    model = LinearRegression()
    model.fit(X, y)
    quarter_models[q] = {
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': model.score(X, y)
    }

    ax.plot(quarter_data['year'], quarter_data['value'], 'o-', markersize=5,
            linewidth=1, label='Данные', color=colors[q - 1])
    ax.plot(quarter_data['year'], model.predict(X), '--',
            linewidth=2, label='Регрессия', color='black')

    ax.set_title(f'{q} квартал', fontsize=14)
    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('Прибыль', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 3. Анализ динамики
most_dynamic = max(quarter_models.items(), key=lambda x: abs(x[1]['slope']))
least_dynamic = min(quarter_models.items(), key=lambda x: abs(x[1]['slope']))

print("Анализ динамики по кварталам:")
print(f"Наибольшая динамика: Q{most_dynamic[0]} (коэф. {most_dynamic[1]['slope']:.4f})")
print(f"Наименьшая динамика: Q{least_dynamic[0]} (коэф. {least_dynamic[1]['slope']:.4f})")

# 4. Прогнозирование
X_all = data[['time']].values
y_all = data['value'].values

# Линейная модель
linear_model = LinearRegression()
linear_model.fit(X_all, y_all)

# Полиномиальная модель (для более точного прогноза)
poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_model.fit(X_all, y_all)

# Прогноз на 2016 год
years_to_predict = np.array([2016 + (q - 1) / 4 for q in range(1, 5)]).reshape(-1, 1)
linear_pred = linear_model.predict(years_to_predict)
poly_pred = poly_model.predict(years_to_predict)

print("\nПрогноз прибыли на 2016 год:")
print("{:<10} {:<15} {:<15}".format("Квартал", "Линейный прогноз", "Полиномиальный прогноз"))
for q in range(4):
    print("Q{:<9} {:<15.2f} {:<15.2f}".format(q + 1, linear_pred[q], poly_pred[q]))

print("\nСреднегодовой прогноз на 2016:")
print(f"Линейная модель: {linear_pred.mean():.2f}")
print(f"Полиномиальная модель: {poly_pred.mean():.2f}")