from numba import autojit, jit, double, int32, int64, float64
from scipy.linalg import inv, pinv, eigh
from joblib import Parallel, delayed
from tensorflow import Session
from math import sqrt, pi, exp
from scipy.stats import norm
import numpy as np



class Cumulants(object):

    def __init__(self,N=[],hMax=100.):
        self.N = N
        self.N_is_list_of_multivariate_processes = all(isinstance(x, list) for x in self.N)
        if self.N_is_list_of_multivariate_processes:
            self.dim = len(self.N[0])
        else:
            self.dim = len(self.N)
        self.L = np.zeros(self.dim)
        if self.N_is_list_of_multivariate_processes:
            self.time = float(max([max([x[-1]-x[0] for x in multivar_process if x is not None and len(x) > 0]) for multivar_process in self.N]))
        else:
            self.time = float(max([x[-1]-x[0] for x in self.N if x is not None and len(x) > 0]))
        self.C = None
        self.L_th = None
        self.C_th = None
        self.R_true = None
        self.mu_true = None
        self.hMax = hMax

    ###########
    ## Decorator to compute the cumulants on each day, and average
    ###########

    def average_if_list_of_multivariate_processes(func):
        def average_cumulants(self,*args,**kwargs):
            if getattr(self, 'N_is_list_of_multivariate_processes', False):
            #if self.N_is_list_of_multivariate_processes:
                for n, multivar_process in enumerate(self.N):
                    cumul = Cumulants(N=multivar_process)
                    res_one_process = func(cumul,*args,**kwargs)
                    if n == 0:
                        res = np.zeros_like(res_one_process)
                    res += res_one_process
                res /= n+1
            else:
                res = func(self,*args,**kwargs)
            return res
        return average_cumulants

    #########
    ## Functions to compute third order cumulant
    #########

    @average_if_list_of_multivariate_processes
    def compute_L(self):
        self.dim = len(self.N)
        L = np.zeros(self.dim)
        for i, process in enumerate(self.N):
            if process is None:
                L[i] = -1.
            else:
                L[i] = len(process) / self.time
        return L

    @average_if_list_of_multivariate_processes
    def compute_C(self,H=0.,method='parallel',weight='constant',sigma=1.0):
        if H == 0.:
            hM = self.hMax
        else:
            hM = H
        d = self.dim
        if method == 'classic':
            C = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    C[i,j] = A_ij(self.N[i],self.N[j],-hM,hM,self.time,self.L[j],weight=weight,sigma=sigma)
        elif method == 'parallel':
            l = Parallel(-1)(delayed(A_ij)(self.N[i],self.N[j],-hM,hM,self.time,self.L[j],weight=weight,sigma=sigma) for i in range(d) for j in range(d))
            C = np.array(l).reshape(d,d)
        # we keep the symmetric part to remove edge effects
        C[:] = 0.5 * (C + C.T)
        return C

    def set_L(self):
        self.L = self.compute_L()

    def set_C(self,H=0.,method='parallel',weight='constant',sigma=1.0):
        self.C = self.compute_C(H=H,method=method,weight=weight,sigma=sigma)

    def set_R_true(self,R_true):
        self.R_true = R_true

    def set_mu_true(self,mu_true):
        self.mu_true = mu_true

    def set_L_th(self):
        assert self.R_true is not None, "You should provide R_true."
        assert self.mu_true is not None, "You should provide mu_true."
        self.L_th = get_L_th(self.mu_true, self.R_true)

    def set_C_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.C_th = get_C_th(self.L_th, self.R_true)

    def set_all(self,H=0.,method="parallel",weight='constant',sigma=1.0):
        self.set_L()
        print("L is computed")
        self.set_C(H=H,method=method,weight=weight,sigma=sigma)
        print("C is computed")
        if self.R_true is not None and self.mu_true is not None:
            self.set_L_th()
            self.set_C_th()

##########
## Theoretical cumulants L, C
##########

@autojit
def get_L_th(mu, R):
    return np.dot(R,mu)

@autojit
def get_C_th(L, R):
    return np.dot(R,np.dot(np.diag(L),R.T))

##########
## Useful fonctions to set_ empirical integrated cumulants
##########
@autojit
def weight_fun(x, sigma, mode='constant'):
    if mode == 'constant':
        return 1.
    elif mode == 'gaussian':
        return sigma * sqrt(2*pi) * norm.pdf(x, scale=sigma)

#@jit(double(double[:],double[:],int32,int32,double,double,double), nogil=True, nopython=True)
#@jit(float64(float64[:],float64[:],int64,int64,int64,float64,float64), nogil=True, nopython=True)
@autojit
def A_ij(Z_i,Z_j,a,b,T,L_j,weight='constant',sigma=1.0):
    """
    Computes the mean centered number of jumps of N^j between \tau + a and \tau + b, that is
    \frac{1}{T} \sum_{\tau \in Z^i} ( N^j_{\tau + b} - N^j_{\tau + a} - \Lambda^j (b - a) )
    """
    res = 0
    u = 0
    n_i = Z_i.shape[0]
    n_j = Z_j.shape[0]
    if weight == 'constant':
        trend_j = L_j*(b-a)
    elif weight == 'gaussian':
        trend_j = L_j*sigma*sqrt(2*pi)*(norm.sf(a)-norm.sf(b))
    for t in range(n_i):
        # count the number of jumps
        tau = Z_i[t]
        if tau + a < 0: continue
        while u < n_j:
            if Z_j[u] <= tau + a:
                u += 1
            else:
                break
        delta = 0.
        v = u
        while v < n_j:
            if Z_j[v] < tau + b:
                delta += weight_fun(Z_j[v]-tau, sigma, mode=weight)
                v += 1
            else:
                break
        if v == n_j: continue
        res += delta-trend_j
    res /= T
    return res
