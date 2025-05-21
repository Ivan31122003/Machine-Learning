import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Загрузка данных
data = pd.read_csv('cars.csv')
X = data[['speed']]  # скорость (мили в час)
y = data['dist']     # тормозной путь (футы)

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Прогноз для 40 миль/час
speed_40 = pd.DataFrame({'speed': [40]})
predicted_dist = model.predict(speed_40)

# Вычисление R²
r_squared = model.score(X, y)

# Вывод результатов
print(f"Тормозной путь при 40 миль/час: {predicted_dist[0]:.2f} футов")
print(f"Коэффициент детерминации R²: {r_squared:.3f}")

# Создание графика
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Наблюдаемые данные')
plt.plot(X, model.predict(X), color='red', linewidth=2,
         label=f'Линейная регрессия (R² = {r_squared:.3f})')
plt.scatter(40, predicted_dist, color='green', s=100, label='Прогноз для 40 миль/час')

# Настройка графика
plt.title('Зависимость тормозного пути от скорости', fontsize=14)
plt.xlabel('Скорость (миль/час)', fontsize=12)
plt.ylabel('Тормозной путь (футы)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Показать график
plt.show()