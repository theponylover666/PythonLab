from sqlalchemy import create_engine, MetaData

# Подключение к базе данных PostgreSQL через SQLAlchemy
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Создаем объект MetaData для работы со схемой базы данных
metadata = MetaData()

# Загружаем информацию о таблицах из базы данных
metadata.reflect(bind=engine)

# Получаем список всех таблиц
tables = metadata.tables
print("Список таблиц в базе данных:")
for table_name in tables:
    print(f"- {table_name}")

# Получаем описание столбцов для каждой таблицы
for table_name, table in tables.items():
    print(f"\nТаблица: {table_name}")
    for column in table.columns:
        print(f"Столбец: {column.name}, Тип данных: {column.type}")
