import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors

# Настройка отображения графиков (используем доступный стиль)
plt.style.use('ggplot')  # Вместо 'seaborn' используем 'ggplot' или другой доступный стиль
plt.rcParams['figure.figsize'] = (15, 5)

# Проверка доступных стилей
print("Доступные стили:", plt.style.available)

# Загрузка данных
datasets = {}
for i in range(1, 4):
    filename = f'clustering_{i}.csv'
    try:
        data = pd.read_csv(filename, sep='\t', header=None)
        datasets[f'Dataset {i}'] = data.values
    except FileNotFoundError:
        print(f"Файл {filename} не найден. Проверьте путь к файлу.")
        exit()


# Функция для визуализации кластеров
def plot_clusters(data, labels, title, ax):
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'k'  # Шум (для DBSCAN)
        mask = labels == label
        ax.scatter(data[mask, 0], data[mask, 1], c=[color],
                   label=f'Cluster {label}' if label != -1 else 'Noise')
    ax.set_title(title)
    ax.legend()


# Обработка каждого набора данных
for name, data in datasets.items():
    print(f"\n{'=' * 50}\nАнализируем {name}\n{'=' * 50}")

    # Стандартизация данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 1. K-means кластеризация
    print("\n1. Метод K-means:")

    # Определение оптимального числа кластеров
    wcss = []
    silhouette_scores = []
    max_clusters = 10

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'{name}: Определение оптимального числа кластеров (K-means)')

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

        if k > 1:
            silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

    # Метод локтя
    ax1.plot(range(1, max_clusters + 1), wcss, 'bo-')
    ax1.set_title('Метод локтя')
    ax1.set_xlabel('Число кластеров')
    ax1.set_ylabel('WCSS')

    # Силуэтный анализ
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'bo-')
    ax2.set_title('Силуэтный анализ')
    ax2.set_xlabel('Число кластеров')
    ax2.set_ylabel('Коэффициент силуэта')
    plt.show()

    # Выбираем оптимальное k (по силуэту)
    optimal_k = np.argmax(silhouette_scores) + 2  # +2 потому что начинаем с k=2
    print(f"Оптимальное число кластеров: {optimal_k}")

    # Кластеризация с оптимальным k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_data)

    # Визуализация
    fig, ax = plt.subplots()
    plot_clusters(scaled_data, kmeans_labels, f'{name}: K-means (k={optimal_k})', ax)
    plt.show()

    # 2. DBSCAN кластеризация
    print("\n2. Метод DBSCAN:")

    # Определение оптимального eps (метод k-расстояний)
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(scaled_data)
    distances, _ = nbrs.kneighbors(scaled_data)
    distances = np.sort(distances[:, -1], axis=0)

    plt.plot(distances)
    plt.title(f'{name}: Метод k-расстояний для определения eps')
    plt.xlabel('Точки')
    plt.ylabel('Расстояние до 5-го соседа')
    plt.show()

    # Подбираем параметры
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    min_samples_values = [3, 4, 5, 6, 7, 8]

    best_score = -1
    best_params = {}

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(scaled_data)

            # Пропускаем если все точки - шум или один кластер
            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(scaled_data, labels)
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_labels = labels

    print(f"Лучшие параметры: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
    print(f"Коэффициент силуэта: {best_score:.4f}")
    print(f"Найдено кластеров: {len(np.unique(best_labels)) - 1}")  # -1 для учета шума

    # Визуализация
    fig, ax = plt.subplots()
    plot_clusters(scaled_data, best_labels,
                  f"{name}: DBSCAN (eps={best_params['eps']}, min_samples={best_params['min_samples']})", ax)
    plt.show()

    # 3. Иерархическая кластеризация
    print("\n3. Иерархическая кластеризация:")

    # Построение дендрограммы
    linked = linkage(scaled_data, 'ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linked)
    plt.title(f'{name}: Дендрограмма')
    plt.xlabel('Индекс точки')
    plt.ylabel('Расстояние')
    plt.show()

    # Определение числа кластеров (по дендрограмме)
    max_d = 3  # Пороговое расстояние (настраивается по дендрограмме)
    hier_labels = fcluster(linked, max_d, criterion='distance')
    n_clusters = len(np.unique(hier_labels))
    score = silhouette_score(scaled_data, hier_labels)

    print(f"Число кластеров: {n_clusters}")
    print(f"Коэффициент силуэта: {score:.4f}")

    # Визуализация
    fig, ax = plt.subplots()
    plot_clusters(scaled_data, hier_labels, f'{name}: Иерархическая кластеризация (k={n_clusters})', ax)
    plt.show()