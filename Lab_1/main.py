import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import psycopg2

# Настроим стиль графиков
sns.set(style="whitegrid")

# Функция для подключения к базе данных и загрузки данных
def load_data():
    # Подключаемся к базе данных
    engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

    # Запросы для получения данных
    query_film = "SELECT * FROM film;"
    query_category = "SELECT category_id, name FROM category;"
    query_film_category = "SELECT film_id, category_id FROM film_category;"

    # Чтение данных в pandas DataFrame
    df_film = pd.read_sql(query_film, engine)
    df_category = pd.read_sql(query_category, engine)
    df_film_category = pd.read_sql(query_film_category, engine)

    # Объединяем данные film и film_category
    df_film_category = pd.merge(df_film, df_film_category, on="film_id", how="inner")

    return df_film, df_category, df_film_category


# Функция для одномерного анализа (построение гистограмм)
def one_dimensional_analysis(df_film):
    # Гистограмма для rental_duration
    plt.figure(figsize=(10, 6))
    sns.histplot(df_film['rental_duration'], kde=True, color='skyblue', bins=20)
    plt.title('Распределение rental_duration (Продолжительность аренды)', fontsize=16)
    plt.xlabel('Продолжительность аренды (дни)', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.show()

    # Гистограмма для rental_rate
    plt.figure(figsize=(10, 6))
    sns.histplot(df_film['rental_rate'], kde=True, color='orange', bins=20)
    plt.title('Распределение rental_rate (Стоимость аренды)', fontsize=16)
    plt.xlabel('Стоимость аренды ($)', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.show()

    # Гистограмма для length
    plt.figure(figsize=(10, 6))
    sns.histplot(df_film['length'], kde=True, color='green', bins=20)
    plt.title('Распределение length (Длительность фильма)', fontsize=16)
    plt.xlabel('Длительность фильма (минуты)', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.show()

    # Гистограмма для replacement_cost
    plt.figure(figsize=(10, 6))
    sns.histplot(df_film['replacement_cost'], kde=True, color='purple', bins=20)
    plt.title('Распределение replacement_cost (Стоимость замены фильма)', fontsize=16)
    plt.xlabel('Стоимость замены ($)', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.show()


# Функция для многомерного анализа (построение графиков зависимости между признаками)
def multi_dimensional_analysis(df_film_category, df_film,df_category):
    # Объединяем df_film_category с df_category по category_id, чтобы добавить названия категорий
    df_film_category = pd.merge(df_film_category, df_category, on='category_id', how='inner')

    # Создаем график: распределение стоимости аренды по категориям и рейтингам
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_film_category, x='name', y='rental_rate', hue='rating', palette='Set3')
    plt.title('Распределение стоимости аренды по категориям и рейтингам', fontsize=16)
    plt.xlabel('Категория фильма', fontsize=12)
    plt.ylabel('Стоимость аренды ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='Рейтинг', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Группируем по name (название категории) и rental_rate (стоимость аренды) и считаем среднюю продолжительность фильмов
    heatmap_data = df_film_category.groupby(['name', 'rental_rate'])['length'].mean().unstack()

    # Строим тепловую карту
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".1f",
                cbar_kws={'label': 'Средняя продолжительность фильма (минуты)'})
    plt.title('Средняя продолжительность фильмов по категориям и стоимости аренды', fontsize=16)
    plt.xlabel('Стоимость аренды ($)', fontsize=12)
    plt.ylabel('Категория фильма', fontsize=12)
    plt.show()


# Основная функция
def main():
    # Загружаем данные
    df_film, df_category, df_film_category = load_data()

    # Вывод первых 5 строк каждой таблицы
    print("Данные из таблицы film:")
    print(df_film.head(), "\n")

    print("Данные из таблицы film_category:")
    print(df_category.head(), "\n")

    print("Объединенные данные:")
    print(df_film_category.head(), "\n")

    # Вывод общей информации о данных
    print("Информация о таблице film:")
    print(df_film.info(), "\n")

    print("Основные статистики таблицы film:")
    print(df_film.describe(), "\n")

    # Одномерный анализ
    one_dimensional_analysis(df_film)

    # Многомерный анализ
    multi_dimensional_analysis(df_film_category, df_film,df_category)

# Запуск основного процесса
if __name__ == "__main__":
    main()
