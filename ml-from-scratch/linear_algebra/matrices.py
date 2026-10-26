import numpy as np

def matrix_multiply(matrix_a, matrix_b):
    if(len(matrix_a[0]) != len(matrix_b)):
            raise RuntimeError("Incompatible for multiplication")
        
    n, am, bm = len(matrix_a), len(matrix_a[0]), len(matrix_b[0])

    res = [[0] * bm for _ in range(n)] 
    for i in range(n):
        for j in range(bm):
            row = matrix_a[i,:]
            col = matrix_b[:,j]
            res[i][j] = np.sum(row * col)
    return res
