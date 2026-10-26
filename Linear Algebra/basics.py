import numpy as np

class Machine_Learning:
    def __init__(self):
        pass
    
    def dot_product(self, a, b):
        if len(a) != len(b):
            raise RuntimeError("Mismatch dimensions")
        
        return np.sum(a * b)
    
    def l2_norm(self, a):
        return np.sqrt(self.dot_product(a, a))
    
    def cosine(self, a, b):
        return self.dot_product(a, b) / (self.l2_norm(a) * self.l2_norm(b))
    
    def mat_mul(self, a, b):
        if(len(a[0]) != len(b)):
            raise RuntimeError("Incompatible for multiplication")
        
        # n, am, bm = len(a), len(a[0]), len(b[0])
        # i, j, k = 0, 0, 0
        # res = [[0] * bm for _ in range(n)] 
        # for i in range(n):
        #     for j in range(bm):
        #         row = a[i,:]
        #         col = b[:,j]
        #         res[i][j] = np.sum(row * col)
        res = a[:, :, np.newaxis] * b[:, :, np.newaxis]
        
        return res
    
ml = Machine_Learning()
a = np.array([2, 3, 4])
b = np.array([5, 6, 7])
c = np.array([5, 6])


# print(ml.dot_product(a, b))
print("a + b", a + b)
print("a * b", a * b)
print("a @ b", a @ b)

# print(ml.dot_product(a, c))
print(ml.l2_norm(a))

print(ml.mat_mul(a, b))