import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv('nsw74psid1.csv')

# Анализ данных
print("Первые 5 строк данных:")
print(data.head())
print("\nОписание данных:")
print(data.describe())
print("\nПропущенные значения:")
print(data.isnull().sum())

# Визуализация распределения целевой переменной
plt.figure(figsize=(10, 6))
sns.histplot(data['re78'], bins=50, kde=True)
plt.title('Распределение целевой переменной re78')
plt.show()

# Разделение на признаки и целевую переменную
X = data.drop('re78', axis=1)
y = data['re78']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Препроцессинг для разных типов признаков
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# Функция для оценки моделей
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('feature_selection', SelectKBest(f_regression, k='all')),
                               ('model', model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.2f}")

    # Визуализация предсказаний
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Предсказания модели {model_name}')
    plt.show()

    return mse, r2, pipeline


# Модели для сравнения
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'SVR': SVR(C=1.0, epsilon=0.2)
}

# Оценка всех моделей
results = {}
for name, model in models.items():
    mse, r2, pipeline = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    results[name] = {'MSE': mse, 'R2': r2, 'pipeline': pipeline}

# Сравнение результатов
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df = results_df.sort_values(by='MSE')
print("\nСравнение моделей:")
print(results_df[['MSE', 'R2']])

# Выбор лучшей модели
best_model_name = results_df.index[0]
best_model = results[best_model_name]['pipeline']
print(f"\nЛучшая модель: {best_model_name}")

# Анализ важности признаков для лучшей модели
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    feature_selector = best_model.named_steps['feature_selection']
    feature_scores = best_model.named_steps['model'].feature_importances_

    # Получаем имена признаков после преобразования
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Для категориальных признаков нужно получить имена после OneHotEncoding
    if len(categorical_features) > 0:
        ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        categorical_features_ohe = ohe.get_feature_names_out(categorical_features)
        all_features = np.concatenate([numeric_features, categorical_features_ohe])
    else:
        all_features = numeric_features

    # Сортируем признаки по важности
    feature_importance = pd.DataFrame({'Feature': all_features[feature_selector.get_support()],
                                       'Importance': feature_scores})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Визуализация важности признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Важность признаков в лучшей модели')
    plt.show()

# Полиномиальные признаки для линейных моделей
print("\nПопробуем добавить полиномиальные признаки для линейных моделей...")

poly_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())])

poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Полиномиальная регрессия 2-й степени:")
print(f"  MSE: {mse_poly:.2f}")
print(f"  R²: {r2_poly:.2f}")

# Сравнение с предыдущими результатами
if mse_poly < results_df.iloc[0]['MSE']:
    print("\nПолиномиальная регрессия показала лучший результат!")
    best_model_name = "Polynomial Regression (degree=2)"
    best_model = poly_pipeline
else:
    print("\nПолиномиальная регрессия не улучшила результат.")

# Финальная оценка лучшей модели
print("\nФинальная оценка лучшей модели:")
if best_model_name == "Polynomial Regression (degree=2)":
    y_pred_best = y_pred_poly
else:
    y_pred_best = best_model.predict(X_test)

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"{best_model_name}:")
print(f"  MSE: {mse_best:.2f}")
print(f"  R²: {r2_best:.2f}")

# Визуализация финальных предсказаний
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title(f'Предсказания лучшей модели ({best_model_name})')
plt.show()

# Сохранение лучшей модели
import joblib

joblib.dump(best_model, 'best_model.pkl')
print("\nЛучшая модель сохранена в файл 'best_model.pkl'")