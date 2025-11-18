from imports import np, plt,lb1,lb3
def gen_Q(A, tol=1e-12):
    n, m = A.shape
    k = min(n, m)
    Q = np.zeros((n, k))
    for i in range(k):
        v = A[:, i].copy()
        for j in range(i):
            inner_product = lb1.vector_dot(Q[:, j], v)
            v = v - inner_product * Q[:, j]
        norm_2 = lb3.norma(v,2)
        if norm_2 > tol:
            Q[:, i] = v / norm_2
        else:
            Q[:, i] = np.zeros(n)

    return Q
    
def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de m x n (con m >= n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A tiene dimensiones m < n, debe retornar None
    """
    m,n = A.shape
    if(m < n):
      return None
  
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for i in range(n):
        v = A[:, i].copy()
        for j in range(i):
            inner_product = lb1.vector_dot(Q[:, j], v)
            R[j,i] = inner_product
            v = v - inner_product * Q[:,j]
        norm_2 = lb3.norma(v,2)
        if norm_2 > tol:
            R[i,i] = norm_2
            Q[:,i] = v/norm_2
        else:
            R[i,i] = 0.0
            Q[:,i] = 0.0

    return Q,R


def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    m,n = A.shape
    if(m < n):
      return None
    Q = np.eye(m)
    R = A.copy()
    for i in range(0, n - 1):
      print(f"Computado: {i+1} de {n-1}", end="\r", flush=True)
      norm_2 = lb3.norma(R[i:, i],2)
      if(norm_2 < tol):
        R[i:, i] = 0;
        continue;
      aux = np.zeros_like(R[i:, i])
      aux[0] = norm_2;
      v =  R[i:, i] - aux;
      u = v / lb3.norma(v,2);
      Hi = Hi = np.eye(m - i) - 2 * lb1.outer(u, u)

      Hi = np.block([[np.eye(i), np.zeros((i, m-i))], [np.zeros((m-i, i)), Hi]])
      R = np.dot(Hi, R);
      Q = np.dot(Q, Hi.T);
    print("Householder completado")
    return Q,R




def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if(metodo == 'RH'):
      print("Iniciando Householder")
      return QR_con_HH(A,tol)
    elif(metodo == 'GS'):
      return QR_con_GS(A,tol)
    else:
      return None
