from numba import autojit, jit, double, int32, int64, float64
from scipy.linalg import inv, pinv, eigh
from joblib import Parallel, delayed
from math import sqrt, pi, exp
from scipy.stats import norm
from itertools import product
import numpy as np


class Cumulants(object):

    def __init__(self, realizations=[], half_width=100.):
        if all(isinstance(x, list) for x in realizations):
            self.realizations = realizations
        else:
            self.realizations = [realizations]
        self.dim = len(self.realizations[0])
        self.n_realizations = len(self.realizations)
        self.time = np.zeros(self.n_realizations)
        for day, realization in enumerate(self.realizations):
            T_day = float(max(x[-1] for x in realization if len(x) > 0)) - float(min(x[0] for x in realization if len(x) > 0))
            self.time[day] = T_day
        self.L = np.zeros((self.n_realizations, self.dim))
        self.C = np.zeros((self.n_realizations, self.dim, self.dim))
        self.L_th = None
        self.C_th = None
        self.R_true = None
        self.mu_true = None
        self.half_width = half_width


    #########
    ## Functions to compute third order cumulant
    #########

    def compute_L(self):
        for day, realization in enumerate(self.realizations):
            L = np.zeros(self.dim)
            for i in range(self.dim):
                process = realization[i]
                if process is None:
                    L[i] = -1.
                else:
                    L[i] = len(process) / self.time[day]
            self.L[day] = L.copy()


    def compute_C(self, half_width=0., method='parallel_by_day', filtr='rectangular', sigma=1.0):
        if half_width == 0.:
            h_w = self.half_width
        else:
            h_w = half_width
        d = self.dim

        if filtr == "rectangular":
            A_ij = A_ij_rect
        elif filtr == "gaussian":
            A_ij = A_ij_gauss
        else:
            raise ValueError("In `compute_C_and_J`: `filtr` should either equal `rectangular` or `gaussian`.")

        if method == 'parallel_by_day':
            l = Parallel(-1)(delayed(worker_day_C)(A_ij, realization, h_w, T, L, sigma, d) for (realization, T, L) in zip(self.realizations, self.time, self.L))
            self.C = [0.5*(z+z.T) for z in l]

        elif method == 'parallel_by_component':
            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                l = Parallel(-1)(
                        delayed(A_ij)(realization[i], realization[j], h_w, self.time[day], self.L[day][j], sigma)
                        for i in range(d) for j in range(d))
                C = np.array(l).reshape(d, d)
                # we keep the symmetric part to remove edge effects
                C[:] = 0.5 * (C + C.T)
                self.C[day] = C.copy()

        elif method == 'classic':
            for day in range(len(self.realizations)):
                realization = self.realizations[day]
                C = np.zeros((d,d))
                for i, j in product(range(d), repeat=2):
                    z = A_ij(realization[i], realization[j], h_w, self.time[day], self.L[day][j], sigma)
                    C[i,j] = z.real
                # we keep the symmetric part to remove edge effects
                C[:] = 0.5 * (C + C.T)
                self.C[day] = C.copy()

        else:
            raise ValueError("In `compute_CJ`: `method` should either equal `parallel_by_day`, `parallel_by_component` or `classic`.")



    def set_R_true(self, R_true):
        self.R_true = R_true

    def set_mu_true(self, mu_true):
        self.mu_true = mu_true

    def set_L_th(self):
        assert self.R_true is not None, "You should provide R_true."
        assert self.mu_true is not None, "You should provide mu_true."
        self.L_th = get_L_th(self.mu_true, self.R_true)

    def set_C_th(self):
        assert self.R_true is not None, "You should provide R_true."
        self.C_th = get_C_th(self.L_th, self.R_true)

    def compute_cumulants(self, half_width=0., method="parallel_by_day", filtr='rectangular', sigma=0.):
        self.compute_L()
        print("L is computed")
        if filtr == "gaussian" and sigma == 0.: sigma = half_width/5.
        self.compute_C_and_J(half_width=half_width, method=method, filtr=filtr, sigma=sigma)
        print("C is computed")
        if self.R_true is not None and self.mu_true is not None:
            self.set_L_th()
            self.set_C_th()


##########
## Theoretical cumulants L, C, K, K_c
##########

@autojit
def get_L_th(mu, R):
    return np.dot(R, mu)


@autojit
def get_C_th(L, R):
    return np.dot(R, np.dot(np.diag(L), R.T))


##########
## Useful fonctions to set_ empirical integrated cumulants
##########


@autojit
def A_ij_rect(realization_i, realization_j, half_width, T, L_j, sigma=1.0):
    """
    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals
    \frac{1}{T} \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} [ (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^j ]
    """
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    res_C = 0
    u = 0
    width = 2 * half_width
    trend_C_j = L_j * width

    for t in range(n_i):
        tau = realization_i[t]
        tau_minus_half_width = tau - half_width

        if tau_minus_half_width < 0: continue

        while u < n_j:
            if realization_j[u] <= tau_minus_half_width:
                u += 1
            else:
                break
        v = u
        sub_res = 0.
        while v < n_j:
            tau_p_minus_tau = realization_j[v] - tau
            if tau_p_minus_tau < half_width:
                sub_res += width - tau_p_minus_tau
                v += 1
            else:
                break
        if v == n_j: continue
        res_C += v - u - trend_C_j
    res_C /= T
    return res_C


@autojit
def A_ij_gauss(realization_i, realization_j, half_width, T, L_j, sigma=1.0):
    """
    Computes the integral \int_{(0,H)} t c^{ij} (t) dt. This integral equals
    \frac{1}{T} \sum_{\tau \in Z^i} \sum_{\tau' \in Z^j} [ (\tau - \tau') 1_{ \tau - H < \tau' < \tau } - H^2 / 2 \Lambda^j ]
    """
    n_i = realization_i.shape[0]
    n_j = realization_j.shape[0]
    res_C = 0
    u = 0
    trend_C_j = L_j * sigma * sqrt(2 * pi) * (norm.cdf(half_width/sigma) - norm.cdf(-half_width/sigma))

    for t in range(n_i):
        tau = realization_i[t]
        tau_minus_half_width = tau - half_width

        if tau_minus_half_width < 0: continue

        while u < n_j:
            if realization_j[u] <= tau_minus_half_width:
                u += 1
            else:
                break
        v = u
        sub_res_C = 0.
        while v < n_j:
            tau_p_minus_tau = realization_j[v] - tau
            if tau_p_minus_tau < half_width:
                sub_res_C += exp(-.5*(tau_p_minus_tau/sigma)**2)
                v += 1
            else:
                break
        if v == n_j: continue
        res_C += sub_res_C - trend_C_j
    res_C /= T
    return res_C


def worker_day_C(fun, realization, h_w, T, L, sigma, d):
    C = np.zeros((d, d))
    for i, j in product(range(d), repeat=2):
        if len(realization[i])*len(realization[j]) != 0:
            z = fun(realization[i], realization[j], h_w, T, L[j], sigma)
            C[i,j] = z.real
    return C

