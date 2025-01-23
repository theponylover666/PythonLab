from sqlalchemy import create_engine
import pandas as pd

# Подключение к базе данных PostgreSQL через SQLAlchemy
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Запросы для получения данных
query_film = "SELECT film_id, title, rental_rate, length FROM film;"
query_category = "SELECT film_id, category_id FROM film_category;"

# Чтение данных в pandas DataFrame
df_film = pd.read_sql(query_film, engine)
df_category = pd.read_sql(query_category, engine)

# Объединение таблиц
df_merged = pd.merge(df_film, df_category, on="film_id", how="inner")

# Печать объединенной таблицы
print(df_merged.head())