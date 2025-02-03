import pandas as pd
import scipy.stats as stats
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Загрузка данных из таблицы film
with engine.connect() as conn:
    df_film = pd.read_sql("SELECT * FROM film", conn)

# Разделим фильмы на короткие и длинные
median_length = df_film['length'].median()
group_short = df_film[df_film['length'] < median_length]['rental_rate'].dropna()
group_long = df_film[df_film['length'] >= median_length]['rental_rate'].dropna()

# Сравнение средних значений rental_rate
mean_short = group_short.mean()
mean_long = group_long.mean()

print(f"Средний rental_rate для коротких фильмов: {mean_short:.2f}")
print(f"Средний rental_rate для длинных фильмов: {mean_long:.2f}")

# Гипотеза проверяется: если средние значения значимо различаются, гипотеза верна
if abs(mean_short - mean_long) > 0.5:  # Условие по усмотрению, можно варьировать порог
    print("Гипотеза верна: Средний rental_rate отличается между короткими и длинными фильмами.")
else:
    print("Гипотеза неверна: Средний rental_rate не отличается.")
