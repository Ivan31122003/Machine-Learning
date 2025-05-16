import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve


# 1. Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Not Survived', 'Survived'])
    plt.yticks([0, 1], ['Not Survived', 'Survived'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


# 2. ROC Curve
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


# 3. Feature Importance (для мета-классификатора)
def plot_feature_importance(model):
    try:
        # Получаем важность признаков из мета-классификатора
        if hasattr(model.named_steps['classifier'].final_estimator_, 'coef_'):
            importance = model.named_steps['classifier'].final_estimator_.coef_[0]
        else:
            importance = model.named_steps['classifier'].final_estimator_.feature_importances_

        features = ['RandomForest', 'SVM', 'k-NN']  # Названия базовых моделей
        plt.figure(figsize=(8, 4))
        plt.barh(features, importance)
        plt.xlabel('Importance')
        plt.title('Meta-classifier Feature Importance (Base Models Weights)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


# 4. Learning Curve
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.show()


# Вызов функций для построения графиков
if 'Survived' in test_data.columns:
    y_test = test_data['Survived']
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    # 1. Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)

    # 2. ROC Curve
    plot_roc_curve(y_test, y_pred_prob)

    # 3. Feature Importance
    plot_feature_importance(model)

    # 4. Learning Curve (на тренировочных данных)
    plot_learning_curve(model, X_train, y_train)
else:
    print("Test labels not available - cannot generate evaluation plots")