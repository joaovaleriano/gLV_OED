#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:37:13 2022

@author: valeriano
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

@njit
def glv(t, x, p):
    n = x.shape[0]
    r = p[:n]
    A = p[n:].reshape((n, n))
    
    return x * (r + np.dot(A, x))

@njit
def glv_jac(t, x, p):
    n = x.shape[0]
    r = p[:n]
    A = p[n:].reshape((n, n))
    
    jac = np.zeros((n, n*(n+1)))
    
    for i in range(n):
        jac[i,i] = x[i]
        
        for j in range(n):
            jac[i,n*(1+i):n*(2+i)] = x[i]*x
    
    return jac


@njit
def euler_maruyama(f, t, x, p, sig, dt, n_steps):
    dt_sqrt = dt**0.5
    n = x.shape[0]
    noise = sig*np.random.randn(n_steps, n)
    
    x_ = np.zeros((n_steps+1, n))
    x_[0] = x
    
    for i in range(1, n_steps+1):
        x_[i] = x_[i-1] + dt * f(t, x_[i-1], p) + dt_sqrt * x_[i-1]*noise[i-1]
        x_[i][x_[i]<0] = 0.
        t += dt
        
    return x_

@njit
def set_nb_seed(seed):
    np.random.seed(seed)
    

#%%

n = 10

np.random.seed(1)
r = np.random.uniform(0, 1, n)
A = np.random.normal(size=(n,n))
for i in range(n):
    A[i,i] = -np.abs(np.random.normal(10))

mat_c = 0
while (-np.linalg.inv(A)@r < 0).any() or (np.linalg.eig(-np.linalg.inv(A)@r.reshape((-1,1))*A)[0]>0).any():
    mat_c += 1
    print(f"new matrix {mat_c}")
    A = np.random.normal(0, size=(n,n))
    for i in range(n):
        A[i,i] = -np.abs(np.random.normal(3, 0.1))

p = np.concatenate((r, A.flatten()))
sig = 0.1 #np.random.uniform(0, 0.1, n)

x0 = np.random.uniform(0, 1, n)

dt = 1e-3
n_steps = 30000

#%%

x = np.zeros((n_steps, n))
t = 0.
set_nb_seed(12)

sol = euler_maruyama(glv, t, x0, p, sig, dt, n_steps)

plt.plot(sol)