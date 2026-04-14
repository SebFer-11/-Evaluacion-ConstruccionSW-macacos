import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 📂 1. Cargar dataset (ajustado a tu carpeta)
df = pd.read_csv("DataSetA/house_prices.csv")  # cambia el nombre si es distinto

# 🔎 2. Verificar datos
print(df.head())

# 📊 3. Correlación con la variable objetivo (precio)
corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
print("\nCorrelaciones con SalePrice:\n", corr)

# 🧠 4. Seleccionar la variable más correlacionada
# (normalmente es GrLivArea, pero puede cambiar según tu dataset)
feature = corr.index[1]  # la segunda es la mejor variable (la primera es SalePrice)

print("\nVariable seleccionada:", feature)

X = df[[feature]]
y = df["SalePrice"]

# 🧠 5. Entrenar modelo
model = LinearRegression()
model.fit(X, y)

# 🔮 6. Predicciones
y_pred = model.predict(X)

# 📉 7. Métricas
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n📊 RESULTADOS:")
print("MSE:", mse)
print("R2:", r2)

# 📈 8. Gráfico
plt.scatter(X, y, color="blue", label="Datos reales")
plt.plot(X, y_pred, color="red", label="Regresión lineal")

plt.xlabel(feature)
plt.ylabel("SalePrice")
plt.title("Regresión Lineal Simple - House Prices")
plt.legend()
plt.show()
