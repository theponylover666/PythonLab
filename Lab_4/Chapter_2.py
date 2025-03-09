import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Загрузка данных
df = pd.read_csv("CarPrice_Assignment.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)
print("Данные загружены. Первые 5 строк:")
print(df.head())

# Первичный анализ
print("\nРазмеры датафрейма:")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

print("\nИспользование памяти:")
memory_usage = df.memory_usage(deep=True).sum() / (1024)
print(f"Используется памяти: {memory_usage:.2f} килобайт")

print("\nСтатистика числовых признаков:")
print(df.describe())

# Обработка данных
df.drop(columns=['car_ID'], inplace=True)
df['CarBrand'] = df['CarName'].apply(lambda x: x.split()[0])
df.drop(columns=['CarName'], inplace=True)

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('price')
for col in numerical_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\nРазмеры датафрейма после обработки выбросов:")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

# Кодирование категориальных переменных
df = pd.get_dummies(df, drop_first=True)

# Разделение данных
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nРазмеры выборок после разбиения:")
print(f"Обучающая: {X_train.shape}, Тестовая: {X_test.shape}")

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.any(y_true != 0) else np.nan
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R²": r2}

models = {
    "Linear Regression": LinearRegression(),
    "LASSO": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5)
}

results = {}
for name, model in models.items():
    print(f"\nОбучение модели: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = evaluate_model(y_test, y_pred)

    # Визуализация предсказаний (ступенчатый график)
    plt.figure(figsize=(8, 5))
    sorted_indices = np.argsort(y_test)
    plt.step(y_test.iloc[sorted_indices], y_pred[sorted_indices], where='mid', label="Предсказания", color='blue')
    plt.plot(y_test.iloc[sorted_indices], y_test.iloc[sorted_indices], 'r--', label="Идеальное предсказание")
    plt.xlabel("Фактические значения")
    plt.ylabel("Предсказанные значения")
    plt.title(f"Ступенчатый график предсказаний для {name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

print("\nРезультаты оценки моделей:")
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")

# Выбор лучшей модели
best_model_name = min(results, key=lambda x: (results[x]["MAE"], results[x]["MSE"], results[x]["RMSE"], -results[x]["R²"]))
best_model = models[best_model_name]
print(f"\nЛучшая модель по совокупности метрик: {best_model_name}")

# Сохранение модели
model_path = "best_model.pkl"
joblib.dump(best_model, model_path)
print(f"Модель сохранена в {model_path}")

# Проверка загрузки модели
loaded_model = joblib.load(model_path)
print("Модель успешно загружена")