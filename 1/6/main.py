import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
train_data = pd.read_csv('bank_scoring_train.csv', sep='\t')
test_data = pd.read_csv('bank_scoring_test.csv', sep='\t')

# 2. Предобработка данных
# Заполнение пропущенных значений медианой
imputer = SimpleImputer(strategy='median')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Разделение на признаки и целевую переменную
X_train = train_data.drop('SeriousDlqin2yrs', axis=1)
y_train = train_data['SeriousDlqin2yrs']
X_test = test_data.drop('SeriousDlqin2yrs', axis=1)
y_test = test_data['SeriousDlqin2yrs']

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Построение и обучение моделей
# Логистическая регрессия
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Случайный лес
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X_train, y_train)

# 4. Оценка качества моделей
# Функция для оценки моделей
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Вероятности для ROC-AUC
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("-" * 50)

# Оценка логистической регрессии
print("Logistic Regression:")
evaluate_model(log_reg, X_test, y_test)

# Оценка случайного леса
print("Random Forest:")
evaluate_model(rf_clf, X_test, y_test)