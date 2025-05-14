import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def reduce_color_palette(image_path, n_colors=16):
    """Сжимает цветовую палитру изображения до n_colors с помощью K-means."""
    image = Image.open(image_path)
    image_array = np.array(image)

    h, w, d = image_array.shape
    pixels = image_array.reshape(-1, d)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    new_image_array = new_colors.reshape(h, w, d).astype(np.uint8)

    return Image.fromarray(new_image_array)


# Загружаем изображение
image_path = "image1.jpg"  # Замените на свой файл
original_image = Image.open(image_path)

# Сжимаем до 4, 8, 16 и 32 цветов
compressed_4 = reduce_color_palette(image_path, 4)
print(1)
compressed_8 = reduce_color_palette(image_path, 8)
print(2)
compressed_16 = reduce_color_palette(image_path, 16)
print(3)

# Создаем полотно с 4 изображениями (оригинал + 3 сжатых)
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.title("Исходное изображение")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("4 цвета")
plt.imshow(compressed_4)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("8 цветов")
plt.imshow(compressed_8)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("16 цветов")
plt.imshow(compressed_16)
plt.axis('off')

plt.tight_layout()
plt.show()