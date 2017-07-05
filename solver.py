from numpy.linalg import eigh
from whma.updates import *
import numpy as np

#@autojit
def admm(cumul, prox_fun, X1_0=None, X4_0=None, rho=0.1, alpha=0.99, maxiter=100, positivity=True):
    """
    ADMM framework to minimize a prox-capable objective over the matrix of kernel norms.
    """

    d = cumul.dim

    if X1_0 is None:
        X1_0 = np.zeros((d,d))
    if X4_0 is None:
        X4_0 = np.eye(d)
    # compute diagA, diagD, O, B and C
    L_mean = np.mean(cumul.L, axis=0)
    C_mean = np.mean(cumul.C, axis=0)
    diagA = np.sqrt(L_mean)
    diagD, O = eigh(C_mean)
    sqrt_diagD = np.sqrt(diagD)
    B = np.dot(O,np.dot(np.diag(sqrt_diagD),O.T))
    C = np.diag(1. / diagA)

    # initialize parameters
    X1 = X1_0.copy()
    X2 = X1_0.copy()
    X3 = X1_0.copy()
    X4 = X4_0.copy()
    Y1 = np.dot(np.diag(1. / diagA), X1_0)
    #Y1 = X1_0.copy()
    Y2 = np.dot(X4_0, np.dot(O,np.dot(np.diag(1. / sqrt_diagD),O.T)))
    #Y2 = X1_0.copy()
    U1 = np.zeros_like(X1_0)
    U2 = np.zeros_like(X1_0)
    U3 = np.zeros_like(X1_0)
    U4 = np.zeros_like(X1_0)
    U5 = np.zeros_like(X1_0)

    for _ in range(maxiter):
        X1[:] = update_X1(prox_fun, X2, Y1, U2, U4, diagA, rho=rho)
        X2[:] = update_X2(X1, X3, U2, U3, positivity)
        X3[:] = update_X3(X2, U3, alpha=alpha)
        X4[:] = update_X4(Y2, U5, B)
        Y1[:] = update_Y1(X1, Y2, U1, U4, diagA, C)
        Y2[:] = update_Y2(X4, Y1, U1, U5, sqrt_diagD, O, B, C)
        U1[:] = update_U1(U1, Y1, Y2, C)
        U2[:] = update_U2(U2, X1, X2)
        U3[:] = update_U3(U3, X2, X3)
        U4[:] = update_U4(U4, X1, Y1, diagA)
        U5[:] = update_U5(U5, X4, Y2, B)

#    print("||X1 - X_2|| = ", np.linalg.norm(X1-X2))
#    print("||X2 - X_3|| = ", np.linalg.norm(X2-X3))
#    print("||U1|| = ", np.linalg.norm(U1))
#    print("||U2|| = ", np.linalg.norm(U2))
#    print("||U3|| = ", np.linalg.norm(U3))
#    print("||U4|| = ", np.linalg.norm(U4))
#    print("||U5|| = ", np.linalg.norm(U5))

    return X1.T
