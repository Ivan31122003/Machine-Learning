import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('votes.csv', index_col=0)

# Обработка пропущенных значений (замена NA на среднее по столбцу)
data_filled = data.fillna(data.mean())

# Нормализация данных
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_filled)

# Построение дендрограммы
plt.figure(figsize=(15, 10))
plt.title('Дендрограмма')
plt.xlabel('Штаты')
plt.ylabel('Расстояние')

# Вычисление связи между кластерами методом Уорда
Z = linkage(data_scaled, method='ward')

# Построение дендрограммы
dendrogram(Z, labels=data.index, leaf_rotation=90, leaf_font_size=8)
plt.tight_layout()
plt.show()