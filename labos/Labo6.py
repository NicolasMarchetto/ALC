from imports import np, plt,lb3,lb1
import cupy as cp

def metpot2k(A, tol=1e-15, K=1000):
    n = A.shape[0]
    vectorResultante = np.random.rand(n)
    vectorResultante = vectorResultante / lb3.norma(vectorResultante, 2)
    cuentas = 0

    for i in range(int(K)):
        Av = lb1.matmulti(A, vectorResultante)
        vectorSiguiente = Av / lb3.norma(Av, 2)
        cuentas += 1

        dif = vectorSiguiente - vectorResultante
        if lb3.norma(dif, 2) < tol:
            break
        vectorResultante = vectorSiguiente

    Av = lb1.matmulti(A, vectorSiguiente)
    autovalor = lb1.vector_dot(vectorSiguiente, Av) / lb1.vector_dot(vectorSiguiente, vectorSiguiente)
    return vectorSiguiente, autovalor, cuentas


def diagRH(A, tol=1e-15, K=1000):
    n = A.shape[0]
    v1, lambda1, k = metpot2k(A, tol, K)

    e1 = np.zeros_like(v1)
    e1[0] = 1

    sign = 1.0 if v1[0] >= 0 else -1.0
    norm_v1 = lb3.norma(v1, 2)
    w = v1 + sign * norm_v1 * e1
    nw = lb3.norma(w, 2)

    if nw < tol:
        H_v1 = np.eye(n)
    else:
        w = w / nw

        H_v1 = np.eye(n)
        for i in range(n):
            for j in range(n):
                H_v1[i, j] -= 2 * w[i] * w[j]

    if n == 1:
        S = H_v1
        D = lb1.matmulti(lb1.matmulti(H_v1, A), H_v1)
        return S, D
    else:
        B = lb1.matmulti(lb1.matmulti(H_v1, A), H_v1)
        A_prima = B[1:, 1:]
        S_prima, D_prima = diagRH(A_prima, tol, K)

        D = np.zeros((n, n))
        D[0, 0] = lambda1
        D[1:, 1:] = D_prima

        S = np.eye(n)
        S[1:, 1:] = S_prima
        S = lb1.matmulti(H_v1, S)
        return S, D


