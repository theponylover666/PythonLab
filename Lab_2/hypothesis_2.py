import pandas as pd
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Подключение к базе данных и загрузка данных с учетом rental_rate
with engine.connect() as conn:
    df_category = pd.read_sql("""
        SELECT f.film_id, f.length, f.rental_rate, c.name AS category
        FROM film f
        JOIN film_category fc ON f.film_id = fc.film_id
        JOIN category c ON fc.category_id = c.category_id
    """, conn)

# Проверяем, какие столбцы есть в датафрейме (на всякий случай)
print("Столбцы в df_category:", df_category.columns)

# Средняя стоимость аренды по жанрам
rental_rate_by_category = df_category.groupby('category')['rental_rate'].mean()

print("Средняя стоимость аренды по категориям:")
print(rental_rate_by_category)

# Проверяем гипотезу
if rental_rate_by_category.std() > 0.5:  # Порог можно изменить
    print("Гипотеза верна: Стоимость аренды зависит от жанра.")
else:
    print("Гипотеза неверна: Стоимость аренды не зависит от жанра.")

