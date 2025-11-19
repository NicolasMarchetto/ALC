from imports import np, plt,lb1
import cupy as cp

def calculaLU(A):
    """
    Calcula la factorización LU de la matriz A con unos en la diagonal de L.
    Retorna (L, U, nops).
    Si no se puede factorizar, retorna (None, None, 0).

    nops cuenta las multiplicaciones y sumas/restas hechas durante la eliminación.
    """
    print("Calculando LU")
    if A is None:
      return None, None, 0

    n , m = A.shape
    if n != m:
      return None, None, 0
    L = np.eye(n)
    U = A.copy()
    nops = -n  #Esto es porque en el programa cuento como una resta poner el 0 y no lo deberia contar

    for k in range(n-1):
        print(f"Calculando fila {k+1} de {n-1}", end="\r", flush=True)
        pivot = U[k, k]
        if np.abs(pivot) < 1e-08: #Si todos los pivotes de U son distintos a 0, det(U) != 0
            return None, None, 0

        for i in range(k+1, n):
            L[i, k] = U[i, k] / pivot
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]
                nops += 2
                
    if np.abs(U[n-1, n-1]) < 1e-08: #Verificamos también para el último pivot
         return None, None, 0

    return L, U, nops


def res_tri(L, b, inferior=True):
    n = L.shape[0]
    b = cp.array(b)
    L = cp.array(L)
    x = cp.zeros(n)

    if inferior:  
        for i in range(n):
            s = b[i]
            # cp.dot evita el bucle interno
            if i > 0:
                s -= cp.dot(L[i, :i], x[:i])
            x[i] = s / L[i, i]
    else: 
        for i in range(n - 1, -1, -1):
            s = b[i]
            if i < n - 1:
                s -= cp.dot(L[i, i+1:], x[i+1:])
            x[i] = s / L[i, i]

    return cp.asnumpy(x)

def inversa(A):
  L,U,nops = calculaLU(A)
  if L is None:
    return None
  else:
    n = A.shape[0]
    inversa =np.zeros((n, n))

    for i in range(n):
      b = np.zeros(n)
      b[i] = 1
      d = res_tri(L, b, inferior=True)
      x = res_tri(U, d, inferior=False)
      inversa[:, i] = x

    return inversa


def diagonal_de_matriz(A):
  x,y = A.shape
  if x != y:
    return None

  nueva_matriz = np.zeros((x,y))
  for i in range(x):
    for j in range(y):
      if j == i:
        nueva_matriz[i,j] = A[i,j]
  return nueva_matriz




def calculaLDV(A):
    print("Calculando LDV")
    n = A.shape[0]
    L,U,nops = calculaLU(A)
    if L is None:
      return None,None,None
    D = np.eye(n)
    for i in range(n):
      print(f"Calculando D fila {i+1} de {n}", end="\r", flush=True)
      D[i][i] = U[i][i]

    for i in range(n):
      print(f"Calculando U fila {i+1} de {n}", end="\r", flush=True)
      for j in range(n):
        U[i, j] = U[i,j] / D[i, i]
    return L,D,U

def esSimetrica(A):

    return lb1.matricesIguales(A, A.T)


def la_diagonal_es_positiva(A):
  return np.all(np.diag(A) > 0)

def esSDP(A,atol=1e-8):
    """
    Checkea si la matriz A es simétrica definida positiva (SDP) usando
    la factorización LDV.
    """
    L,D,V = calculaLDV(A)
    print("Verificando si A es SDP")
    iguales = True
    if L is None:
        return False
    for i in range(A.shape[0]):
      print(f"Verificando simetría fila {i+1} de {A.shape[0]}", end="\r", flush=True)
      for j in range(A.shape[1]):
        if not lb1.sonIguales(A[i,j], lb1.transpuesta(A)[i,j],atol):
          iguales = False
    return iguales and la_diagonal_es_positiva(D)
