import pandas as pd
import statsmodels.api as sm
from tabulate import tabulate

# Загрузка данных
data = pd.read_csv('reglab1.txt', delim_whitespace=True)

# Функция для построения и оценки модели
def build_model(target, predictors):
    X = sm.add_constant(data[predictors])
    model = sm.OLS(data[target], X).fit()
    return model

# Строим три модели
models = {
    "x ~ y + z": build_model('x', ['y', 'z']),
    "y ~ x + z": build_model('y', ['x', 'z']),
    "z ~ x + y": build_model('z', ['x', 'y'])
}

# Выводим результаты
results = []
for name, model in models.items():
    results.append([
        name,
        model.rsquared,
        model.aic,
        model.bic,
        *[f"{p:.4f}" for p in model.pvalues[1:]]  # p-значения для предикторов
    ])  # Исправлено: закрывающая квадратная скобка

# Красивое сравнение моделей
headers = ["Модель", "R²", "AIC", "BIC", "p-value 1", "p-value 2"]
print(tabulate(results, headers=headers, floatfmt=".3f"))

# Детальный вывод каждой модели
for name, model in models.items():
    print(f"\n\nДетальный отчет для модели {name}:")
    print(model.summary())

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    sns.regplot(x=model.predict(), y=data[name.split(' ')[0]], ax=ax)
    ax.set_title(f"{name}\nR²={model.rsquared:.3f}")
    ax.set_xlabel("Предсказанные значения")
    ax.set_ylabel("Наблюдаемые значения")
plt.tight_layout()
plt.show()