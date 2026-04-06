import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import functions

with open("parameters.json", "r", encoding="utf-8") as file:
    params = json.load(file)

data = pd.read_csv('insurance.csv')

alpha = params["alpha"]
num_iters = params["num_iters"]

data["sex"] = data["sex"].map({"male":1,"female":0})
data["smoker"] = data["smoker"].map({"yes":1,"no":0})

X = data[["age","sex","bmi","children","smoker"]].copy()
Y = data["charges"].copy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    train_size=0.8,
    random_state=42,
    shuffle=True,
    stratify=None
)

mean_values = X_train.mean()
std_values = X_train.std()

X_train_norm = (X_train - mean_values) / std_values
X_test_norm = (X_test - mean_values) / std_values

X_train = X_train_norm.values
X_test = X_test_norm.values

Y_train= Y_train.values.reshape(-1,1)
Y_test= Y_test.values.reshape(-1,1)

m = X_train.shape[0]
X_train = np.c_[np.ones(m), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

theta = np.zeros((X_train.shape[1],1))

#trening
theta, J_history = functions.gradient_descent(X_train, Y_train, theta, alpha, num_iters)

#results
print(f"Theta: \n{theta}")

#final cost
final_cost = functions.cost(X_train, Y_train, theta)
print(f"\nFinal cost: {final_cost}")

#predictions
predictions = functions.linear_regression(X_test, theta)

print("\nPredictions (first 5):")
print(predictions[:5])

print("\nReal values (first 5):")
print(Y_test[:5])

#shapes
print("\nShapes:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)