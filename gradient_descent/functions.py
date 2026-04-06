import numpy as np
import json

with open("parameters.json", "r", encoding="utf-8") as file:
    data = json.load(file)

def linear_regression(X, theta):
    return np.dot(X, theta)

def cost(X, Y, theta):
    m = X.shape[0]
    h = np.dot(X, theta)
    J = (1 / (2 * m)) * np.sum((h - Y) ** 2)
    return J

def gradient_descent(X, Y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        predictions = np.dot(X, theta)
        errors = predictions - Y

        gradient = (1 / m) * np.dot(X.T, errors)
        theta = theta - alpha * gradient

        J_history[i] = cost(X, Y, theta)

    return theta, J_history
