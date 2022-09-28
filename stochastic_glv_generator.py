#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:37:13 2022

@author: valeriano
"""

# packages

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#%%
# functions

@njit
def glv(t, x, p):
    """
    glv: right-hand-side of generalized Lotka-Volterra model ODE describing N interacting species

    --- INPUT ---
    t: time points to evaluate function. scalar
    x: species abundances. array (N,)
    p: model parameters. array (N*(N+1),)
       -> p[:N]: growth rates
       -> p[N:]: flattened interaction matrix

    --- OUTPUT ---
    rhs: right-hand-side of gLV
    """

    n = x.shape[0]
    r = p[:n]
    A = p[n:].reshape((n, n))
    
    rhs = x * (r + np.dot(A, x))

    return rhs


@njit
def glv_jac(t, x, p):
    """
    glv_jac: jacobian of generalized Lotka-Volterra model ODE describing N interacting species

    --- INPUT ---
    t: time points to evaluate function. scalar
    x: species abundances. array (N,)
    p: model parameters. array (N*(N+1),)
       -> p[:N]: growth rates
       -> p[N:]: flattened interaction matrix

    --- OUTPUT ---
    jac: jacobian matrix of the gLV system for given state (x) and parameters (p)
    """

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
    """
    euler_maruyama: Euler-Maruyama method for approximate solution of stochastic differential equations (SDEs)
    N = dimension of the system

    --- INPUT ---
    f: rhs of SDE. function f(t, x, p)
    t: initial time. scalar
    x: initial state of the system. array (N,)
    p: parameters of the model. array
    sig: constant scale multiplying noise. scalar or array (N,)
    dt: size of time step for SDE integration. scalar
    n_steps: number of steps to integrate along. scalar

    --- OUTPUT ---
    x_: time-series generated as solution of the SDE
    """
    
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


def sort_glv_params(n, seed, r_max, A_diag_std, A_off_diag_std):
    """
    sort_glv_params: sorts parameters that lead to a stable gLV system
    growth rates are sorted uniformly between 0 and r_max
    interspecies interactions are sorted normally around zero with A_off_diag_std standard deviation
    intraspecies interactions are sorted as negative absolute values of variables normally around zero with A_diag_std standard deviation

    --- INPUT ---
    n: number of species. scalar
    seed: random seed to initialize parameters
    r_max: maximum growth rate
    A_diag_std: standard deviation for distribution of intraspecies interactions
    A_off)diag_std: standard deviation for distribution of interspecies interactions

    --- OUTPUT ---
    p: chosen parameter set
    """

    np.random.seed(seed)
    r = np.random.uniform(0, r_max, n)
    A = np.random.normal(size=(n,n))
    for i in range(n):
        A[i,i] = -np.abs(np.random.normal(10))

    mat_c = 0
    while (-np.linalg.inv(A)@r < 0).any() or (np.linalg.eig(-np.linalg.inv(A)@r.reshape((-1,1))*A)[0]>0).any():
        mat_c += 1
        print(f"\rnew matrix {mat_c}", end="")
        A = np.random.normal(0, size=(n,n))
        for i in range(n):
            A[i,i] = -np.abs(np.random.normal(3, 0.1))

    p = np.concatenate((r, A.flatten()))

    return p