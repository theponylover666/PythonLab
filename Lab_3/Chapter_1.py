import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    roc_curve

# Загрузка данных
file_path = "train.csv"
df = pd.read_csv(file_path)

# Первичный анализ
print("Количество строк и столбцов:", df.shape)
print("Объем памяти (МБ):", df.memory_usage(deep=True).sum() / (1024 ** 2))

# Выделение числовых признаков
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T

# Анализ пропущенных значений
print("Пропущенные значения:")
print(df.isnull().sum())

# Анализ выбросов
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Удаление выбросов
df_cleaned = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]

# Разделение данных
X = df_cleaned.drop(columns=['id'])  # Исключаем ID
y = X.pop(X.columns[-1])  # Предположим, что последний столбец — целевая переменная

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Нормализация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Определение моделей
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True)
}

# Обучение и оценка моделей
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Доля правильных предсказаний": accuracy_score(y_test, y_pred),
        "Точность": precision_score(y_test, y_pred, average="binary"),
        "Полнота": recall_score(y_test, y_pred, average="binary"),
        "Среднее между точностью и полнотой": f1_score(y_test, y_pred, average="binary"),
        "Качество классификации": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "Таблица ошибок": confusion_matrix(y_test, y_pred)
    }
    results[name] = metrics

# Вывод результатов
results_df = pd.DataFrame(results).T
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)
print(results_df)

# Визуализация матриц ошибок для каждой модели
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"Матрица ошибок - {name}")
    axes[i].set_xlabel("Предсказанный класс")
    axes[i].set_ylabel("Фактический класс")

plt.tight_layout()
plt.show()

# График распределения целевой переменной
plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette="viridis", legend=False)
plt.title("Распределение классов в целевой переменной")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.grid(True)
plt.show()

# График важности признаков для логистической регрессии
log_reg = models["Logistic Regression"]
feature_importance = abs(log_reg.coef_).flatten()
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=feature_names, hue=feature_names, palette="coolwarm", legend=False)
plt.title("Важность признаков для логистической регрессии")
plt.xlabel("Влияние")
plt.ylabel("Признаки")
plt.grid(True)
plt.show()

# ROC-кривые для моделей
plt.figure(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):  # Только модели, имеющие predict_proba
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Линия случайного угадывания
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые моделей")
plt.legend()
plt.grid(True)
plt.show()
