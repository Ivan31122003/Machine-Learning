import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = pd.read_csv('pluton.csv')
print("Первые 5 строк данных:")
print(data.head())

max_iter_list = [5, 10, 15, 20]
results_no_scale = {}
results_scaled = {}

print("\nКластеризация без стандартизации:")
for max_iter in max_iter_list:
    kmeans = KMeans(n_clusters=3, max_iter=max_iter, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    silhouette = silhouette_score(data, clusters)
    results_no_scale[max_iter] = silhouette
    print(f"max_iter={max_iter}: Silhouette Score = {silhouette:.4f}")

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print("\nКластеризация со стандартизацией:")
for max_iter in max_iter_list:
    kmeans = KMeans(n_clusters=3, max_iter=max_iter, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_scaled)
    silhouette = silhouette_score(data_scaled, clusters)
    results_scaled[max_iter] = silhouette
    print(f"max_iter={max_iter}: Silhouette Score = {silhouette:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(list(results_scaled.keys()), list(results_scaled.values()), marker='o', label='Со стандартизацией')
plt.plot(list(results_no_scale.keys()), list(results_no_scale.values()), marker='o', label='Без стандартизации')
plt.xlabel('Максимальное число итераций')
plt.ylabel('Коэффициент силуэта')
plt.title('Сравнение качества кластеризации')
plt.legend()
plt.grid()
plt.show()