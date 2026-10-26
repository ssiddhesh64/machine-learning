import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_loss(w, b, points):
    
    loss = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        loss += (y - (w * x + b)) ** 2
    
    return loss / n

def gradient_descent(w, b, points, L):
    
    w_grad, b_grad = 0, 0
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        w_grad += -(2 / n) * x * (y - (w * x + b))
        b_grad += -(2 / n) * (y - (w * x + b))
    
    w = w - (w_grad) * L
    b = b - (b_grad) * L
    
    return w, b

def plot(x, y, loss):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    axes[0].set_title("Salary vs Experience")
    axes[0].set_xlabel("Experience")
    axes[0].set_ylabel("Salary")
    axes[0].scatter(x, y)

    axes[1].plot(loss)

    plt.tight_layout()
    plt.show()

points = pd.read_csv('salary_data.csv')

x = points['YearsExperience']
y = points['Salary']

epochs = 1000
L = 1e-3

w = 0
b = 0
loss = []
for i in range(epochs):
    loss.append(get_loss(w, b, points))
    w, b = gradient_descent(w, b, points, L)
    
plot(x, y, loss)
    

    
    
