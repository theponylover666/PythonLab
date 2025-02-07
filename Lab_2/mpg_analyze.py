import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import ttest_ind, pearsonr

# 1. Загрузка данных
mpg_df = sns.load_dataset("mpg")

# 2. Количество строк и столбцов
num_rows, num_cols = mpg_df.shape
print(f"Количество строк: {num_rows}, Количество столбцов: {num_cols}")

# 3a. Анализ числовых переменных
numeric_cols = mpg_df.select_dtypes(include=["number"])
numeric_analysis = pd.DataFrame({
    "Пропуски (%)": numeric_cols.isnull().mean() * 100,
    "Максимум": numeric_cols.max(),
    "Минимум": numeric_cols.min(),
    "Среднее": numeric_cols.mean(),
    "Медиана": numeric_cols.median(),
    "Дисперсия": numeric_cols.var(),
    "Квантиль 0.1": numeric_cols.quantile(0.1),
    "Квантиль 0.9": numeric_cols.quantile(0.9),
    "Квартиль 1": numeric_cols.quantile(0.25),
    "Квартиль 3": numeric_cols.quantile(0.75),
})

print("\nАнализ числовых переменных:")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)
print(numeric_analysis)

# 3b. Анализ категориальных переменных
categorical_cols = mpg_df.select_dtypes(include=["object"])
categorical_analysis = pd.DataFrame({
    "Пропуски (%)": categorical_cols.isnull().mean() * 100,
    "Уникальные значения": categorical_cols.nunique(),
    "Мода": categorical_cols.mode().iloc[0]
})

print("\nАнализ категориальных переменных:")
print(categorical_analysis)

# 4. Формулирование гипотез

# Гипотеза 1: Автомобили с 4 цилиндрами экономичнее, чем с 8 цилиндрами
mpg_4cyl = mpg_df[mpg_df["cylinders"] == 4]["mpg"].dropna()
mpg_8cyl = mpg_df[mpg_df["cylinders"] == 8]["mpg"].dropna()
t_stat, p_value = ttest_ind(mpg_4cyl, mpg_8cyl)
print("\nГипотеза 1: Автомобили с 4 цилиндрами экономичнее 8-цилиндровых.")
print(f"t-статистика: {t_stat:.2f}, p-значение: {p_value:.5f}")

# Гипотеза 2: Есть ли связь между мощностью (horsepower) и расходом топлива (mpg)?
mpg_clean = mpg_df.dropna(subset=["mpg", "horsepower"])
corr_coef, p_value_corr = pearsonr(mpg_clean["horsepower"], mpg_clean["mpg"])
print("\nГипотеза 2: Связь между мощностью и расходом топлива.")
print(f"Коэффициент корреляции: {corr_coef:.2f}, p-значение: {p_value_corr:.5f}")

# 5. Кодирование категориальных переменных
selected_categorical_cols = ["origin", "cylinders"]
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = encoder.fit_transform(mpg_df[selected_categorical_cols].fillna("Unknown"))
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out())

# Объединяем закодированные данные с остальными
mpg_encoded = pd.concat([mpg_df.drop(columns=selected_categorical_cols), categorical_encoded_df], axis=1)
mpg_encoded = mpg_encoded.select_dtypes(include=['number'])
print("Закодированные данные")
print(mpg_encoded)

# 6. Корреляционная матрица
correlation_matrix = mpg_encoded.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Корреляционная матрица признаков")
plt.show()


# 7. Реализация градиентного спуска
class GradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Добавляем bias
        self.theta = np.zeros(X.shape[1])
        m = len(y)

        for _ in range(self.epochs):
            gradients = (1 / m) * X.T.dot(X.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.theta)


# Подготовка данных для градиентного спуска
mpg_df = mpg_df.dropna(subset=["mpg", "horsepower", "weight"])
X = mpg_df[["horsepower"]].values
y = mpg_df["mpg"].values

# Обучение модели
model = GradientDescent(learning_rate=0.0001, epochs=10000)
model.fit(X, y)

# Визуализация
plt.scatter(X, y, label="Фактические данные")
plt.plot(X, model.predict(X), color="red", label="Линейная регрессия")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.title("Градиентный спуск для предсказания MPG по Horsepower")
plt.show()
