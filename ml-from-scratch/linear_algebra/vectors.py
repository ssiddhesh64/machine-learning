import numpy as np

def dot_product(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
            raise RuntimeError("Mismatch dimensions")
        
    return np.sum(vector_a * vector_b)

def l1_norm(vector):
    return np.sum(np.abs(vector))

def l2_norm(vector):
    return np.sqrt(np.sum(vector ** 2))
