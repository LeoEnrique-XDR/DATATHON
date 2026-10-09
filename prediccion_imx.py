import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 1. Cargar el dataset original o limpio
data = pd.read_csv("layer2.csv")

# 2. Filtrar únicamente la fila correspondiente a ImmutableX
imx_row = data[data['name'].str.contains('immutable', case=False, na=False)]

# 3. Extraer todas las columnas que corresponden a fechas (contienen '/')
date_cols = [c for c in data.columns if '/' in c]

# Transponer las fechas para tener una serie temporal ordenada
imx_prices = imx_row[date_cols].T
imx_prices.columns = ['price']
imx_prices.index = pd.to_datetime(imx_prices.index, format='%m/%d/%y', errors='coerce')
imx_prices = imx_prices.sort_index().dropna()

# 4. Preparar los datos para el modelo predictivo (Regresión Lineal)
imx_prices['days'] = (imx_prices.index - imx_prices.index[0]).days
X = imx_prices[['days']]
y = imx_prices['price']

# Entrenar el modelo de regresión
model = LinearRegression()
model.fit(X, y)

# 5. Generar predicciones para los siguientes 30 días
last_day = X['days'].max()
future_days_num = np.array([[last_day + i] for i in range(1, 31)])
future_preds = model.predict(future_days_num)

# Crear un índice de fechas futuras para la visualización
future_dates = pd.date_range(start=imx_prices.index[-1] + pd.Timedelta(days=1), periods=30)

# 6. Visualizar el histórico y la tendencia predictiva
plt.figure(figsize=(12, 6))
plt.plot(imx_prices.index, imx_prices['price'], label='Precio Histórico (ImmutableX)', color='blue', marker='o')
plt.plot(future_dates, future_preds, label='Tendencia Predictiva (Próximos 30 días)', color='orange', linestyle='--', marker='x')

plt.title('Análisis Predictivo de Precios - ImmutableX')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# --- Resumen Estadístico (Similar a la tabla de la imagen) ---
print("\n--- RESUMEN ESTADÍSTICO DE IMMUTABLEX ---")

# Seleccionar únicamente la columna de precios y aplicar describe()
resumen_estadistico = imx_prices[['price']].describe()

# Mostrar la tabla formateada en la terminal
print(resumen_estadistico)