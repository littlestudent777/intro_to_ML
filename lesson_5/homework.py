import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')
data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]

X = data_df[["sepal_length", "petal_length"]]
Y = data_df["species"]

data_df_setosa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

# Обучение модели на всех данных
model = SVC(kernel="linear", C=10000)
model.fit(X, Y)
support_vectors = model.support_vectors_

# Создаем новый набор данных, содержащий только опорные векторы
sv_indices = model.support_
X_sv = X.iloc[sv_indices]
Y_sv = Y.iloc[sv_indices]

# Обучаем новую модель на этом наборе
model_sv = SVC(kernel="linear", C=10000)
model_sv.fit(X_sv, Y_sv)


x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
    columns=["sepal_length", "petal_length"]
)

# Предсказание на полном наборе данных
y_p_full = model.predict(X_p)
X_p_full = X_p.copy()
X_p_full["species"] = y_p_full

# Предсказание только на опорных векторах
y_p_sv = model_sv.predict(X_p)
X_p_sv = X_p.copy()
X_p_sv["species"] = y_p_sv

plt.figure(figsize=(14, 8))

# Визуализация с изначальными данными
plt.subplot(1, 2, 1)
plt.scatter(X_p_full[X_p_full["species"] == "setosa"]["sepal_length"],
            X_p_full[X_p_full["species"] == "setosa"]["petal_length"],
            alpha=0.2, color='blue')
plt.scatter(X_p_full[X_p_full["species"] == "versicolor"]["sepal_length"],
            X_p_full[X_p_full["species"] == "versicolor"]["petal_length"],
            alpha=0.2, color='green')
plt.scatter(X["sepal_length"], X["petal_length"],
            c=Y.map({"setosa": "blue", "versicolor": "green"}))
# Опорные вектора помечены
plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
            s=200, facecolors='none', edgecolors='black', linewidths=1.0)
plt.title("Original data with support vectors")
plt.xlabel("sepal_length")
plt.ylabel("petal_length")


# Визуализация только с опорными векторами
plt.subplot(1, 2, 2)
plt.scatter(X_p_sv[X_p_sv["species"] == "setosa"]["sepal_length"],
            X_p_sv[X_p_sv["species"] == "setosa"]["petal_length"],
            alpha=0.2, color='blue')
plt.scatter(X_p_sv[X_p_sv["species"] == "versicolor"]["sepal_length"],
            X_p_sv[X_p_sv["species"] == "versicolor"]["petal_length"],
            alpha=0.2, color='green')
plt.scatter(X_sv["sepal_length"], X_sv["petal_length"],
            c=Y_sv.map({"setosa": "blue", "versicolor": "green"}))
plt.scatter(X_sv["sepal_length"], X_sv["petal_length"],
            s=200, facecolors='none', edgecolors='black', linewidths=1.0)
plt.title("Only support vectors")
plt.xlabel("sepal_length")
plt.ylabel("petal_length")

plt.tight_layout()
plt.savefig('hw_result.png')
