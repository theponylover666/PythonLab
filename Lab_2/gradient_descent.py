import pandas as pd
from sqlalchemy import create_engine

# Подключение к базе данных Sakila
engine = create_engine("postgresql+psycopg2://mpi:135a1@povt-cluster.tstu.tver.ru:5432/sakila_postgres")

# Загрузка данных из таблицы film
with engine.connect() as conn:
    df_film = pd.read_sql("SELECT * FROM film", conn)

# Градиентный спуск для зависимости rental_rate от length
X = df_film.dropna(subset=['length', 'rental_rate'])['length'].values
y = df_film.dropna(subset=['length', 'rental_rate'])['rental_rate'].values
def stochastic_gradient_descent(X, y, learning_rate=0.0001, n_epochs=50):
    m = len(y)
    theta0 = 0
    theta1 = 0
    for epoch in range(n_epochs):
        for i in range(m):
            xi = X[i]
            yi = y[i]
            prediction = theta0 + theta1 * xi
            error = prediction - yi
            theta0 -= learning_rate * error
            theta1 -= learning_rate * error * xi
    return theta0, theta1

theta0_stoch, theta1_stoch = stochastic_gradient_descent(X, y)
print("Стохастический градиентный спуск: Перехват =", theta0_stoch, ", Коэффициент =", theta1_stoch)
