import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv("diabetes.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)
print("Данные загружены. Первые 5 строк:")
print(df.head())

print("\nРазмеры датафрейма:")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

print("\nИспользование памяти:")
memory_usage = df.memory_usage(deep=True).sum() / (1024)  # в килобайтах
print(f"Используется памяти: {memory_usage:.2f} килобайт")

print("\nСтатистика числовых признаков:")
print(df.describe())

print("\nМоды категориальных переменных:")
for col in df.columns:
    mode_val = df[col].mode()[0]
    mode_count = (df[col] == mode_val).sum()
    print(f"{col}: мода = {mode_val}, встречается {mode_count} раз")

columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)

print("\nКоличество пропусков в данных после замены нулей:")
print(df.isnull().sum())

df[columns_to_replace] = df[columns_to_replace].fillna(df[columns_to_replace].median())

for col in df.columns:
    if col != "Outcome":
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\nРазмеры датафрейма после обработки выбросов:")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\nОбнаружены категориальные переменные, требуется кодирование:")
    print(categorical_cols)
else:
    print("\nВ наборе данных нет категориальных переменных, кодирование не требуется.")

target = "Outcome"
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nРазмеры выборок после разбиения:")
print(f"Обучающая: {X_train.shape}, Тестовая: {X_test.shape}")

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Исправленный расчет RMSE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.any(y_true != 0) else np.nan
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R²": r2}

models = {
    "Linear Regression": LinearRegression(),
    "LASSO": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    print(f"\nОбучение модели: {name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = evaluate_model(y_test, y_pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='black', label="data", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="идеальное предсказание")

    plt.xlabel("Фактические значения", fontsize=12)
    plt.ylabel("Предсказанные значения", fontsize=12)
    plt.title(f"График предсказаний для {name}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

print("\nРезультаты оценки моделей:")
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")

best_model_name = min(results,
                      key=lambda x: (results[x]["MAE"], results[x]["MSE"], results[x]["RMSE"], -results[x]["R²"]))
best_model = models[best_model_name]
print(f"\nЛучшая модель по совокупности метрик: {best_model_name}")

model_path = "best_model.pkl"
joblib.dump(best_model, model_path)
print(f"Модель сохранена в {model_path}")

loaded_model = joblib.load(model_path)
print("Модель успешно загружена")
