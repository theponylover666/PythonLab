import pandas as pd
import scipy.stats as stats
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Загрузка данных из таблицы film
with engine.connect() as conn:
    df_film = pd.read_sql("SELECT * FROM film", conn)

# Гипотеза 1: Средний rental_rate различается между фильмами разной длительности
median_length = df_film['length'].median()
group_short = df_film[df_film['length'] < median_length]['rental_rate'].dropna()
group_long = df_film[df_film['length'] >= median_length]['rental_rate'].dropna()
t_stat, p_value = stats.ttest_ind(group_short, group_long)
print("Гипотеза 1: Различие среднего rental_rate между короткими и длинными фильмами")
print(f"t-статистика: {t_stat}, p-значение: {p_value}")
