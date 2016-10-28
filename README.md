# When Hawkes Met ADMM

We use the same *Cumulants* object than in NPHC (without third order cumulant related functions).

The idea is to minimize a prox-capable function ```f (||\Phi||)``` given three constraints:
* the nonnegativity of ```||\Phi||```
* the stability of the Hawkes process
* constraint from the integrated covariance

The minimization is done via the ADMM algorithm.
