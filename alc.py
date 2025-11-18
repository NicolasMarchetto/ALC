from imports import *
from labos.Labo1 import matmulti, transpuesta
from labos.Labo4 import inversa
import cupy as cp
"""
Módulo: alc.py
Trabajo Práctico - Álgebra Lineal Computacional
------------------------------------------------
Implementaciones manuales de funciones matriciales
utilizando las rutinas desarrolladas en el laboratorio.

IMPORTANTE:
    - NO usar funciones de numpy.linalg (inv, pinv, svd, qr, cholesky, etc.)
    - Usar únicamente las funciones desarrolladas por el grupo.
    - Todas las funciones deben respetar la forma matricial solicitada.

Autores:

-Franco V. Grasso 132/23
-Ramiro Arbetman 1307/24
- Agregar
- Agregar

Materia: Álgebra Lineal Computacional (UBA)
"""

from imports import np, plt
import os

####################################
# 1. LECTURA DE DATOS
####################################


def cargarDataset(basePath):
    """
    Carga el dataset completo de gatos y perros a partir de un path
    
    La función espera que dentro de basePath existan dos subcarpetas principales:
    - train: con las carpetas cats y dogs que contienen los embeddings de entrenamiento.
    - val: con las carpetas cats y dogs que contienen los embeddings de validación.

    Internamente, llama a cargarCarpeta para procesar cada par de carpetas (gatos y perros)
    y devuelve las matrices con los datos y etiquetas correspondientes.
    
    Retorna:
        Xt, Yt -> datos y etiquetas de entrenamiento
        Xv, Yv -> datos y etiquetas de validación
    """
    trainCats = os.path.join(basePath, "train", "cats")
    trainDogs = os.path.join(basePath, "train", "dogs")
    valCats   = os.path.join(basePath, "val", "cats")
    valDogs   = os.path.join(basePath, "val", "dogs")

    Xt, Yt = cargarCarpeta(trainCats, trainDogs)
    Xv, Yv = cargarCarpeta(valCats, valDogs)

    return Xt, Yt, Xv, Yv

def cargarCarpeta(pathGatos, pathPerros):
    """
    Carga los embeddings de gatos y perros desde dos carpetas dadas y construye las matrices X e Y.
    
    - Recorre todos los archivos .npy de ambas carpetas.
    - Cada archivo se carga como un vector columna .
    - Las etiquetas (Y) se asignan como:
        * [1, 0] para gatos
        * [0, 1] para perros
    
    Retorna:
        X -> matriz de embeddings
        Y -> matriz de etiquetas 
    """
    XList = []
    YList = []


    for archivo in os.listdir(pathGatos):
        if archivo.endswith(".npy"):
            embedding = np.load(os.path.join(pathGatos, archivo))
            XList.append(embedding)
            aux = np.array([[1, 0]] * embedding.shape[1])
            YList.append(aux)

    for archivo in os.listdir(pathPerros):
        if archivo.endswith(".npy"):
            embedding = np.load(os.path.join(pathPerros, archivo))
            XList.append(embedding)
            aux = np.array([[0, 1]] * embedding.shape[1])
            YList.append(aux)

    X =  np.array(XList)
    Y = np.array(YList)

    return np.hstack((X[0][:,], X[1][:,])), np.vstack((Y[0][:,], Y[1][:,]))


####################################
# 2. ECUACIONES NORMALES
####################################

def pinvEcuacionesNormales(X, Y):
    """
    Calcula los pesos W usando pseudo-inversa por ecuaciones normales y Cholesky.

    X: matriz de entrada (n x p)
    Y: matriz de tipos de imagenes (m x p)

    Devuelve:
    W: pesos (m x n)
    """
    n, p = X.shape
    rangoX = cp.linalg.matrix_rank(cp.array(X))
    print("Rango de X:", rangoX)

    if rangoX == p and n > p:
        XTX = lb1.matmulti(lb1.transpuesta(X), X)
        print("Inicia Cholesky")
        print("XTX shape:",XTX.shape)
        L = cp.linalg.cholesky(cp.array(XTX))
        LT = L.T
        print("L shape:",L.shape)
        print("Finaliza Cholesky")
        XT = lb1.transpuesta(X)
        print("XT shape: ", XT.shape)
        U = np.zeros_like(XT)
        print("U shape: ", U.shape)
        print("Y shape: ", Y.shape)
        for col in range(n):
            print("Columna ", col)
            b = XT[:, col]
            z = lb4.res_tri(L, b, inferior=True)
            u = lb4.res_tri(LT, z, inferior=False)
            U[:, col] = u
    
        W = lb1.matmulti(Y, U)

    elif rangoX == n and n < p:
        XXT = lb1.matmulti(X, lb1.transpuesta(X))
        print("XXT shape:",XXT.shape)
        print("Inicia Cholesky")
        L = cp.linalg.cholesky(cp.array(XXT))
        print("L shape:",L.shape)
        LT = L.T
        print("Finaliza Cholesky")
        XT = lb1.transpuesta(X)
        print("XT shape: ", X.shape)
        V = np.zeros_like(X)
        print("V shape: ", V.shape)
        print("Y shape: ", Y.shape)
        for col in range(n):
            print("Columna ", col)
            b = X[:, col]
            z = lb4.res_tri(L, b, inferior=True)
            v = lb4.res_tri(LT, z, inferior=False)
            V[:, col] = v
        
        W = lb1.matmulti(Y, transpuesta(V))

    elif rangoX == n and n == p:
        XInv = lb4.inversa(X)
        W = lb1.matmulti(Y, XInv)

    return W


####################################
# 3. DESCOMPOSICIÓN EN VALORES SINGULARES (SVD)
####################################

def pinvSVD(U, S, V, Y):
    """
    Calcula los pesos W utilizando la pseudo-inversa obtenida por SVD.

    Parámetros:
        U, S, V : matrices de la descomposición SVD propia
        Y       : matriz de targets de entrenamiento

    Retorna:
        W = Y @ V @ S⁻¹ @ U.T
    """

    #En la funcion nos dan directamente la descomposción.
    #Habria que preguntar si te pasan V o V*
    #Asumo que me pasan V
    v_transpuesta = V.T #Hago transpuesta para pasarla como parametro a calcularPseudoInversa luego

    V, S_inversa, U_transpuesta = calcularPseudoInversa(U, S, v_transpuesta)

    W = calculo_W_SVD(V, S_inversa, U_transpuesta, Y)

    return W



####################################
# 4. DESCOMPOSICIÓN QR
####################################

def pinvHouseHolder(Q, R, Y):
    """
    Calcula los pesos W usando la descomposición QR (Householder).

    Parámetros:
        Q, R : matrices obtenidas de la factorización QR (Householder)
        Y    : matriz de targets

    Retorna:
        W : pesos que minimizan ||Y - W X||²
    """
    # Q R = X.T con  X de dim {n x p}, n < p por lo tanto X.T es de dim {p x n} y Q es de dim {p x p} y R de dim {p x n}
    # Como R es de dim {p x n} con p < n, R no es invertible. por lo tanto se reduce las dimensiones de R.T tal que R sea cuadrada
    n = R.shape[1]
    R_reducida = R[:n, :]  # R de dim {p x n} -> R_reducida de dim {n x n}
    Q_reducida = Q[:, :n]  # Q de dim {p x p} -> Q_reducida de dim {p x n}

    print('R shape: ', R.shape," -> R reducida shape: ", R_reducida.shape)
    print('Q shape: ', Q.shape," -> Q reducida shape: ", Q_reducida.shape)
 
    R_T = transpuesta(R_reducida)  # R.T de dim {p x p}
    R_T_inv = np.zeros_like(R_reducida)  # (R.T)^-1 de dim {p x p}

    print('Q reducida shape x R transpuesta inversa shape: ', Q_reducida.shape," x ", R_T_inv.shape)
    print('W shape: ', Y.shape," x (", Q_reducida.shape[0],",", R_T_inv.shape[1],")")

    for fila in range(R_T.shape[0]):

        print(f'Fila {fila} / {R_T.shape[0]}', end='\r', flush=True)

        R_T_inv[fila, :] = lb4.res_tri(R_T, np.eye(R_T.shape[0])[fila, :], inferior=True)
    
    X_plus = lb1.matmulti(Q_reducida, R_T_inv)  # X+ de dim {p x n}
    W = lb1.matmulti(Y, X_plus)
    print('W shape: ', W.shape)
    print('X_plus shape: ', X_plus.shape)
    return W, X_plus


def pinvGramSchmidt(Q, R, Y):
    """
    Calcula los pesos W usando la descomposición QR (Gram-Schmidt clásico).

    Parámetros:
        Q, R : matrices obtenidas con Gram-Schmidt
        Y    : matriz de targets

    Retorna:
        W : pesos óptimos
    """
    # Como Gram-Schmidt se puede hacer únicamente con una matriz en ℝ^{n×p} con n >= p, hacemos la desc QR en X.T
    # Si QR = X.T -> X+ = Q (R.T)-1 -> X+ R.T = Q
    # Si X.T es de dims {pxn} -> Q es de {pxn} y R es de {nxn} (al igual que (R.T)-1)
    # Luego X+ tiene dims {pxn}
    # Vamos el sistema triangular X+ R.T = Q, con R.T triangular inferior, con cada fila de Q para encontrar todas las filas de X+
    p, n = Q.shape
    pX = np.zeros((p, n))

    for fila in range(p):
        pX[fila, :] = lb4.res_tri(transpuesta(R), Q[fila, :])

    # Calculamos W = YX+
    # Si Y es de dims {2xp}, y X+ de {pxn} -> dim (W) = {2xn}
    W = lb1.matmulti(Y, pX)

    return W


####################################
# 5. PSEUDO-INVERSA DE MOORE-PENROSE
####################################

def esPseudoInversa(X, pX, tol=1e-8):
    """
    Verifica si pX es la pseudo-inversa de X según Moore-Penrose.

    X: matriz original
    pX: matriz candidata a pseudo-inversa

    Devuelve True si cumple las 4 condiciones, False si no.
    """

    # Condición 1: X * pX * X ≈ X
    if not lb1.sonIguales(lb1.matmulti(lb1.matmulti(X , pX) , X), X, tol):
        return False

    # Condición 2: pX * X * pX ≈ pX
    if not lb1.sonIguales(lb1.matmulti(lb1.matmulti(pX ,X) , pX), pX, tol):
        return False

    # Condición 3: (X * pX)^T ≈ X * pX
    if not lb1.sonIguales(lb1.transpuesta(lb1.matmulti(X,pX)), lb1.matmulti(X,pX), tol):
        return False

    # Condición 4: (pX * X)^T ≈ pX * X
    if not lb1.sonIguales(lb1.transpuesta(lb1.matmulti(pX,X)), lb1.matmulti(pX,X), tol):
        return False

    return True

####################################
# 6. EVALUACIÓN Y BENCHMARKING
####################################

def evaluarModelo(W, Xv, Yv):
    """
    Evalúa el modelo sobre el conjunto de validación.
    Devuelve la matriz de confusión y el accuracy.
    """
    Y_pred = W @ Xv
    preds = np.argmax(Y_pred, axis=0)
    reales = np.argmax(Yv, axis=0)

    # Matriz de confusión 2x2
    conf = np.zeros((2, 2), dtype=int)
    for i in range(len(preds)):
        conf[reales[i], preds[i]] += 1

    accuracy = np.trace(conf) / np.sum(conf)
    return conf, accuracy


####################################
# 7. SÍNTESIS FINAL
####################################
# (Esto no es código, sino análisis. Cada integrante redacta su parte en el informe)


####################################
# 0. Funciones auxiliares
####################################

def descCholesky(A):
    """
    Calcula la descomposición de Cholesky A = L L^T
    usando la LDV si A es simétrica definida positiva.
    """
    Lprima, D, _ = lb4.calculaLDV(A)

    if not lb4.esSDP(A):
        return None, None

    if Lprima is None or D is None:
        return None, None

    raizD = np.zeros_like(D)
    for i in range(D.shape[0]):
        raizD[i, i] = np.sqrt(D[i, i])

    L = lb1.matmulti(Lprima, raizD)

    LT = lb1.transpuesta(L)

    return L, LT

def rango(A):
    """
    Retorna el rango de una matriz a partir de la cantidad de valores
    singulares distintos de 0
    """
    _,D,_ = lb8.svd_reducida(A)
    rango = sum(1 for d in D if d > 0)
    return rango


def calcularPseudoInversa(U,S,V_transpuesta):
    """
    Entrada: Descomposicion de valores singulares de una matriz
    Retorna V,S+,U transpuesta que representa la pseudoInversa de A
    """

    V = V_transpuesta.T
    U_transpuesta = U.T
    S_matriz = np.zeros((U.shape[0], V.shape[0]))
    np.fill_diagonal(S_matriz, S)
    S = S_matriz
    print("S shape: ", S.shape)
    filas, columnas = S.shape
    S_inversa = np.zeros((columnas, filas))

    #Uso tolerancia para no hacer divisiones de numeros muy chiquitos
    #Consultar si se tiene que hacer asi a priori
    tol = 1e-12
    for i in range(min(filas, columnas)):
        if S[i, i] > tol:
            S_inversa[i, i] = 1 / S[i, i]


    # A^+ = V * S^+ * U^T
    return V, S_inversa, U_transpuesta




def calculo_W_SVD(V,S_inversa,U_transpuesta, Y):
   # El propósito de esta función es comprender el motivo por el cual resulta conveniente
   # emplear la forma "reducida" de la descomposición SVD en el cálculo de la pseudoinversa.
   # Hasta este punto disponemos de las matrices V, S_inversa, U_transpuesta y Y sin ningún tipo de reducción.
   # El objetivo es ir simplificando gradualmente la representación, conservando solo
   # la información relevante para el cálculo de W.

   # 1) Cálculo del rango de la matriz S_inversa:
   # El rango de S_inversa coincide con la cantidad de valores singulares distintos de cero de X.
   # Este valor es fundamental, ya que determina cuántas direcciones (o componentes) aportan información útil.
   # A partir del rango, podemos realizar las particiones necesarias en U y V para
   # limitar las operaciones únicamente a las dimensiones efectivamente informativas.
   #
   # Por ejemplo, si U es una matriz de 1000x1000 pero X tiene solo 100 valores singulares no nulos,
   # no resulta eficiente operar con las 1000 columnas: basta con conservar las primeras 100,
   # que contienen toda la información relevante. El resultado obtenido es el mismo,
   # pero con un costo computacional mucho menor.
   #
   # De esta manera, la pseudoinversa puede expresarse como:
   #
   #     X⁺ = V Σ⁺ Uᵀ  ≈  V₁ Σ₁⁻¹ U₁ᵀ
   #
   # donde las matrices V₁, Σ₁ y U₁ corresponden a la forma reducida,
   # es decir, solo las componentes asociadas a valores singulares distintos de cero.


    rango = 0
    filas_u, columnas_u = np.shape(S_inversa)
    for i in range(filas_u):
        for j in range(columnas_u):
            if (S_inversa[i,j] > 0):
                rango = rango + 1

    #Ahora la variable rango es la cantidad de elementos > 0 de la diagonal sigma.

    # 2) Partición de las matrices U y V según el rango:
    # Tomamos únicamente las primeras 'rango' columnas o filas necesarias.

    V1 = V[:, :rango] # p×r
    U1_T = U_transpuesta[:rango, :]  #r×n
    S1_inversa = S_inversa[:rango, :rango] #Te queda rxr


    # 3) Justificación de dimensiones:
    # Sabemos que X ∈ ℝ^{n×p}  ⇒  X⁺ ∈ ℝ^{p×n}.
    #
    # La forma reducida de la pseudoinversa es:
    #     X⁺ = V₁ Σ₁⁻¹ U₁ᵀ
    #
    # Analizando las dimensiones:
    #     V₁ Σ₁⁻¹ U₁ᵀ  ⇒  (p×r)(r×r)(r×n)
    #  Multiplicando de derecha a izquierda:
    #     (r×r)(r×n) = (r×n)
    #     (p×r)(r×n) = (p×n)
    #
    #  Por lo tanto, X⁺ ∈ ℝ^{p×n}, cumpliendo con las dimensiones esperadas.
    #
    # Este chequeo asegura la compatibilidad matricial y evita errores
    # de producto al construir la pseudoinversa.

    #Luego,

    #Si el rango efectivo de X es menor que su rango máximo, la pseudoinversa puede expresarse en forma reducida como X⁺ = V₁ Σ₁⁻¹ U₁ᵀ,
    # utilizando únicamente las componentes asociadas a valores singulares no nulos.”

    #Las U2 y V2 no me importan porque son las particiones que se multiplican por los 0's de sigma. No es relevante. Sumaria costo computacional a la funcion

    # 4) Cálculo de la pseudoinversa reducida:
    #     X⁺ = V₁ Σ₁⁻¹ U₁ᵀ
    producto = lb1.matmulti(V1, S1_inversa)          # V₁ Σ₁⁻¹
    pseudo_inv = lb1.matmulti(producto, U1_T) # V₁ Σ₁⁻¹ U₁ᵀ


    print("pseudo_inv shape: ", pseudo_inv.shape)
    # 5) Cálculo final de W:
    #     W = Y X⁺ = Y (V₁ Σ₁⁻¹ U₁ᵀ)
    W = lb1.matmulti(transpuesta(Y), pseudo_inv)                   # Y * (V₁ Σ₁⁻¹ U₁ᵀ)

    return W







