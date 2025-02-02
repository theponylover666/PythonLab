import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Загрузка данных из таблицы film
with engine.connect() as conn:
    df_film = pd.read_sql("SELECT * FROM film", conn)

# Кодирование категориальных переменных
film_encoded = pd.get_dummies(df_film, columns=['rating'], drop_first=True)

# Отбираем только числовые столбцы для корреляции
numeric_cols = film_encoded.select_dtypes(include=['float64', 'int64'])

# Корреляционная матрица
corr_matrix = numeric_cols.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')
plt.show()
