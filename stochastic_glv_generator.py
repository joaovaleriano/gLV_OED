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