from sklearn.preprocessing import LabelEncoder
from functions import load_data
from functions import do_variability_datasets
from functions import create_plot

file_path = 'spam.txt'
X, y = load_data(file_path, 'float')

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

selection, precision = do_variability_datasets(X, y, 4601);

create_plot(selection, precision, 'Зависимость точности модели от размера обучающей выборки', 'Выборка', 'Точность модели')

