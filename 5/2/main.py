import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка данных
data = pd.read_csv('reglab.txt', delim_whitespace=True)
y = data['y']
X = data.drop('y', axis=1)


def find_best_subset(X, y, max_features=None):
    """
    Алгоритм полного перебора для выбора оптимального подмножества признаков
    """
    if max_features is None:
        max_features = X.shape[1]

    best_subset = None
    best_rss = np.inf
    results = []

    for k in range(1, max_features + 1):
        for combo in combinations(X.columns, k):
            X_subset = X[list(combo)]
            model = LinearRegression().fit(X_subset, y)
            y_pred = model.predict(X_subset)
            rss = ((y - y_pred) ** 2).sum()

            results.append({
                'subset': combo,
                'RSS': rss,
                'R2': model.score(X_subset, y),
                'size': k
            })
            print(round(model.score(X_subset, y), 4))

            if rss < best_rss:
                best_rss = rss
                best_subset = combo

    return pd.DataFrame(results), best_subset, best_rss


# Запуск алгоритма
results_df, best_subset, best_rss = find_best_subset(X, y)

# Вывод результатов
print("Лучшее подмножество признаков:", best_subset)
print("Минимальная RSS:", best_rss)

# Анализ результатов
print("\nТоп-5 комбинаций по RSS:")
print(results_df.sort_values('RSS').head(5).to_string(index=False))

# Визуализация
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for size, group in results_df.groupby('size'):
    plt.scatter(group['RSS'], group['subset'].astype(str), label=f'{size} признаков')
plt.axvline(x=best_rss, color='r', linestyle='--')
plt.xlabel('RSS (Остаточная сумма квадратов)')
plt.ylabel('Комбинация признаков')
plt.title('Зависимость RSS от выбранных признаков')
plt.legend()
plt.tight_layout()
plt.show()