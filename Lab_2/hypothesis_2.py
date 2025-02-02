import pandas as pd
import scipy.stats as stats
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Загрузка данных из таблицы film
with engine.connect() as conn:
    df_film = pd.read_sql("SELECT * FROM film", conn)

df_cleaned = df_film[['length', 'rental_rate']].dropna()
corr_coef, p_value_corr = stats.pearsonr(df_cleaned['length'], df_cleaned['rental_rate'])
print("Гипотеза 2: Корреляция между length и rental_rate")
print(f"Коэффициент корреляции: {corr_coef}, p-значение: {p_value_corr}")