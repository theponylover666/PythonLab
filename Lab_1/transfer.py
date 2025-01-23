import sqlite3
import pandas as pd
from sqlalchemy import create_engine
sqlite_conn = sqlite3.connect("C:/Users/Дмитрий/.cache/kagglehub/datasets/atanaskanev/sqlite-sakila-sample-database/versions/1/sqlite-sakila.db")
tables = sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
# Настраиваем соединение
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")
# Перенос данных
for table_name in tables:
    table_name = table_name[0]
    df = pd.read_sql_query(f"SELECT * FROM {table_name};", sqlite_conn)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Таблица {table_name} перенесена.")