import pandas as pd
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Загрузка данных из таблицы film
with engine.connect() as conn:
    df_film = pd.read_sql("SELECT * FROM film", conn)

# Анализ числовых переменных
num_cols = ['rental_duration', 'rental_rate', 'length', 'replacement_cost']
num_stats = {}
for col in num_cols:
    num_stats[col] = {
        "Доля пропусков": df_film[col].isna().sum() / len(df_film),
        "Максимальное значение": df_film[col].max(),
        "Минимальное значение": df_film[col].min(),
        "Среднее значение": df_film[col].mean(),
        "Медиана": df_film[col].median(),
        "Дисперсия": df_film[col].var(),
        "0.1 квантиль": df_film[col].quantile(0.1),
        "0.9 квантиль": df_film[col].quantile(0.9),
        "1 квартиль": df_film[col].quantile(0.25),
        "3 квартиль": df_film[col].quantile(0.75)
    }
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)
print(pd.DataFrame(num_stats))
