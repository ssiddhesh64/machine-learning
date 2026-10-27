import numpy as np

x = np.array([2, 3, 4])
w = np.array([5, 6, 7])

res1 = np.dot(x, w)
res2 = x @ w
res3 = np.multiply(x, w)

print(res1, res2, res3)

x = np.array([1, 2, 3])
w = np.array([4, 5])

# gives dimension error
# print("x @ w", x @ w) 

x = np.array([1, 2, 3])
w = np.array([4, 5, 6])
print("x * w", x * w) 
print("x @ w", x @ w) 


# Three features, four samples
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [2, 3, 4]
])

w = np.array([2, -1, 3])

result = X @ w
result2 = X @ w
print(X, w, result)
print(X.shape, w.shape, result.shape)

print(X, w, result2)
print(X.shape, w.shape, result2.shape)

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

result = A @ B

print(A @ B)
print(np.matmul(A, B))
print(np.dot(A, B))
print(A * B)

# =======================
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

y = np.array([10, 20, 30])

w = np.array([1, 2])
b = 0

y_pred = X @ w + b
error = y_pred - y
dw = (2 / len(X)) * (X.T @ error)
# dw = (2 / len(X)) * np.dot(X.T, error)
db = (2 / len(X)) * np.sum(error)

print(w, dw, b, db)
print(error, y_pred)