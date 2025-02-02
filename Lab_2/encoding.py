import pandas as pd
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Загрузка данных из таблицы film
with engine.connect() as conn:
    df_film = pd.read_sql("SELECT * FROM film", conn)

# Кодирование категориальных переменных
film_encoded = pd.get_dummies(df_film, columns=['rating'], drop_first=True)

print(film_encoded.head())
