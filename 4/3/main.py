import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)

# Загрузка данных
train_data = pd.read_csv('titanic_train.csv')
test_data = pd.read_csv('titanic_test.csv')


# Предобработка данных
def preprocess_data(df, is_train=True):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived'] if is_train else None

    categorical_features = ['Sex', 'Embarked']
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor, X, y


# Подготовка данных
preprocessor, X_train, y_train = preprocess_data(train_data, is_train=True)
_, X_test, _ = preprocess_data(test_data, is_train=False)

# Создание модели стекинга
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='linear', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='auto'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacking_model)
])

# Обучение модели
model.fit(X_train, y_train)

# Разделим тренировочные данные для валидации
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Предсказания на валидационной выборке
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

# Вычисление точности
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Визуализация
plt.figure(figsize=(15, 10))

# 1. Confusion Matrix
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Не выживут', 'Выживут'],
            yticklabels=['Не выживут', 'Выживут'])
plt.title(f'Матрица ошибок\nТочность: {accuracy:.4f}')
plt.ylabel('True')
plt.xlabel('Predict')

# 2. ROC Curve
plt.subplot(2, 2, 2)
fpr, tpr, _ = roc_curve(y_val, y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

# 3. Feature Importance
plt.subplot(2, 2, 3)
try:
    feature_importances = model.named_steps['classifier'].final_estimator_.coef_[0]
    features = ['RandomForest', 'SVM', 'k-NN']
    plt.barh(features, feature_importances)
    plt.xlabel('Importance')
    plt.title('Meta-classifier Feature Weights')
except Exception as e:
    plt.text(0.5, 0.5, 'Feature weights not available', ha='center')
    plt.title('Feature Weights Not Available')

# Предсказание на тестовых данных
predictions = model.predict(X_test)
result = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
result.to_csv('titanic_predictions.csv', index=False)
print("\nPredictions saved to titanic_predictions.csv")

plt.tight_layout()
plt.show()