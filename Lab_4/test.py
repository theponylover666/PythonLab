import joblib
import numpy as np

model = joblib.load("best_model.pkl")
new_data = np.array([[3, 120, 70, 25, 100, 30.0, 0.5, 25]])

prediction = model.predict(new_data)
print(f"Предсказанное значение: {prediction[0]}")