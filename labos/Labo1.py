from imports import np, plt
import cupy as cp

def error(x, y):
    y64 = np.float64(y)   # convertir y a float64
    return abs(x - y64)
def error_relativo(x, y):

    x64 = np.float64(x)
    y64 = np.float64(y)
    return abs(x64 - y64) / abs(x64)

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x, y), 0, atol=atol)

def matricesIguales(unaMatriz,otraMatriz):
  for i in range(unaMatriz.shape[0]):
    for j in range(unaMatriz.shape[1]):
      if not sonIguales(unaMatriz[i,j], otraMatriz[i,j]):
        return False
  return True
def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x, y), 0, atol=atol)

def matmulti(A, B):
    """
    Multiplicación de matriz por matriz o matriz por vector
    """
    # if  len(B.shape) == 1:  # vector
    #     n, m = A.shape
    #     result = np.zeros(n)
    #     for i in range(n):
    #       result[i] = sum(A[i, j] * B[j] for j in range(m))
    #     return result
    # else:  # matriz
    #     n, m = A.shape
    #     m2, p = B.shape
    #     result = np.zeros((n, p))
    #     for i in range(n):
    #       for j in range(p):
    #         result[i, j] = sum(A[i, k] * B[k, j] for k in range(m))
    #     return result
    return cp.asnumpy(cp.dot(cp.array(A), cp.array(B)))

def vector_dot(v, w):
    """
    Producto interno de dos vectores
    """
    v = np.array(v)
    w = np.array(w)
    suma = sum(v[i] * w[i] for i in range(len(v)))
    return suma

def outer(v, w):
    """
    Calcula el producto externo entre dos vectores : 
    devuelve una matriz donde M[i,j] = v[i] * w[j]
    """
    v = np.array(v)
    w = np.array(w)
    n = len(v)
    m = len(w)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = v[i] * w[j]
    return M

def transpuesta(A):
    filas, columnas = A.shape
    ATranspuesta = np.zeros((columnas, filas))
    for i in range(filas):
        for j in range(columnas):
            ATranspuesta[j, i] = A[i, j]
    return ATranspuesta
