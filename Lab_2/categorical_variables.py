import pandas as pd
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Загрузка данных из таблицы film
with engine.connect() as conn:
    df_film = pd.read_sql("SELECT * FROM film", conn)

# Анализ категориальных переменных
cat_cols = ['title', 'rating', 'special_features']
cat_stats = {}
for col in cat_cols:
    cat_stats[col] = {
        "Доля пропусков": df_film[col].isna().sum() / len(df_film),
        "Количество уникальных значений": df_film[col].nunique(),
        "Мода": df_film[col].mode()[0] if not df_film[col].mode().empty else None
    }
print(pd.DataFrame(cat_stats))
