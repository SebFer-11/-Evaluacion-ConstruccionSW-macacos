import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# 1. Cargar datos
data = pd.read_csv("ingresos pacientes  - ingresos.csv")

print("Columnas:")
print(data.columns)

# 2. Limpieza
data = data.drop(columns=[
    'Número de Paciente', 
    'DNI', 
    'Nombre del Paciente',
    'Fecha de Cita',
    'Fecha de Cita.1'
], errors='ignore')

data = data.dropna()

# 3. Variables
y = data['tipo tratamiento numerico']
X = data.drop(columns=['tipo tratamiento numerico'])

# 4. Convertir texto a números
X = pd.get_dummies(X)

# 5. División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Predicción
y_pred = model.predict(X_test)

# 9. Resultados
print("\nResultados:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))

# 10. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()