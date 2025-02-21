import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Подключение к базе данных и загрузка данных
file_path = "WineQT.csv"
df = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)

# Разведочный анализ данных
print(f"Количество строк: {df.shape[0]}, Количество столбцов: {df.shape[1]}")
print(f"Размер датафрейма в памяти: {df.memory_usage(deep=True).sum():} байт")
print("Статистики по интервальным переменным:")
print(df.describe(percentiles=[0.25, 0.5, 0.75]).T)

# Анализ моды для категориальных переменных
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if not categorical_cols:
    print("В данных отсутствуют категориальные переменные.")
else:
    for col in categorical_cols:
        mode_value = df[col].mode()[0]
        mode_count = df[col].value_counts()[mode_value]
        print(f"Мода для {col}: {mode_value} (встречается {mode_count} раз)")

# Обработка пропусков
missing_values = df.isnull().sum()
print("\nПропущенные значения в данных:")
print(missing_values[missing_values > 0])

df.dropna(inplace=True)
print(f"\nПосле удаления пропусков осталось {df.shape[0]} строк.")

# Обнаружение выбросов с помощью IQR
def detect_outliers(df, columns):
    outliers_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_info[col] = len(outliers)
    return outliers_info

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outliers = detect_outliers(df, numerical_cols)

print("\nКоличество выбросов по каждому признаку:")
for col, count in outliers.items():
    print(f"{col}: {count}")

# Удаление выбросов
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print(f"\nПосле удаления выбросов осталось {df.shape[0]} строк.")

# Выбираем два числовых столбца для модели
feature_cols = ['pH', 'alcohol']

if not all(col in df.columns for col in feature_cols):
    raise ValueError(f"Ошибка: один из столбцов {feature_cols} отсутствует в данных!")

X = df[feature_cols].values
y = (df['quality'] > df['quality'].median()).astype(int)

# Разделение на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Реализация KNN и логистической регрессии без библиотек
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            neighbors = np.argsort(distances)[:self.k]
            labels = self.y_train[neighbors]
            predictions.append(Counter(labels).most_common(1)[0][0])
        return np.array(predictions)

class LogisticRegressionCustom:
    def __init__(self, lr=0.1, epochs=5000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            z = np.dot(X, self.theta)
            predictions = self.sigmoid(z)
            gradient = np.dot(X.T, (predictions - y)) / y.size
            self.theta -= self.lr * gradient

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.theta)) >= 0.5).astype(int)

# Обучение моделей
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
log_reg = LogisticRegressionCustom(lr=0.1, epochs=5000)
log_reg.fit(X_train, y_train)

# Оценка качества моделей
def evaluate_model(model, X_test, y_test, name):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)

    print(f"\nМетрики для {name}:")
    print(f"Точность (A): {accuracy:.4f}")
    print(f"Точность положительного класса (P): {precision:.4f}")
    print(f"Полнота (R): {recall:.4f}")
    print(f"Среднее между точностью и полнотой (E): {f1:.4f}")
    print(f"Оценки качества: {roc_auc:.4f}")
    print("Матрица ошибок:")
    print(conf_matrix)

evaluate_model(knn, X_test, y_test, "KNN")
evaluate_model(log_reg, X_test, y_test, "Logistic Regression")

# Функция визуализации границ решений
def plot_decision_boundary(model, X, y, title):
    h = .1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title(title)
    plt.show()

# ROC-кривые для логистической регрессии
y_train_probs_log_reg = log_reg.sigmoid(np.dot(X_train, log_reg.theta))  # Вероятности предсказаний
y_test_probs_log_reg = log_reg.sigmoid(np.dot(X_test, log_reg.theta))

fpr_train_log, tpr_train_log, _ = roc_curve(y_train, y_train_probs_log_reg)
fpr_test_log, tpr_test_log, _ = roc_curve(y_test, y_test_probs_log_reg)

# ROC-кривые для KNN (преобразуем предсказания в вероятности)
y_train_probs_knn = knn.predict(X_train)  # KNN выдаёт бинарные метки
y_test_probs_knn = knn.predict(X_test)

fpr_train_knn, tpr_train_knn, _ = roc_curve(y_train, y_train_probs_knn)
fpr_test_knn, tpr_test_knn, _ = roc_curve(y_test, y_test_probs_knn)

# Визуализация ROC-кривых
plt.figure(figsize=(8, 6))

# Логистическая регрессия
plt.plot(fpr_train_log, tpr_train_log, color="green", linestyle="--", label="ROC Train (LogReg)")
plt.plot(fpr_test_log, tpr_test_log, color="blue", linestyle="--", label="ROC Test (LogReg)")

# KNN
plt.plot(fpr_train_knn, tpr_train_knn, color="darkgreen", label="ROC Train (KNN)")
plt.plot(fpr_test_knn, tpr_test_knn, color="darkblue", label="ROC Test (KNN)")

# Линия случайного классификатора
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend()
plt.show()

# Визуализация границ решений
plot_decision_boundary(knn, X_train, y_train, "KNN Decision Boundary")
plot_decision_boundary(log_reg, X_train, y_train, "Logistic Regression Decision Boundary")
