#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Generating and simulating generalized Lotka-Volterra systems.

Also a function calculating the system's jacobian (glv_jac), for possible gradient descent optimization purposes.
"""


#%%
# packages

import numpy as np
from numba import njit


#%%
# functions

@njit
def set_nb_seed(seed):
    np.random.seed(seed)
    

@njit
def glv(t, x, p):
    """
    glv: right-hand-side of generalized Lotka-Volterra model ODE describing N interacting species

    --- INPUT ---
    t: time point to evaluate function. scalar
    x: species abundances. array (N,)
    p: model parameters. array (N*(N+1),)
       -> p[:N]: growth rates
       -> p[N:]: flattened interaction matrix

    --- OUTPUT ---
    rhs: right-hand-side of gLV. array (N,)
    """

    n = x.shape[0]
    r = p[:n]
    A = p[n:].reshape((n, n))
    
    rhs = x * (r + np.dot(A, x))

    return rhs


@njit
def glv_time(t, x, p):
    """
    glv: right-hand-side of generalized Lotka-Volterra model ODE describing N interacting species, for multiple time points.

    --- INPUT ---
    t: time points to evaluate function. array (M,)
    x: species abundances. array (M, N,)
    p: model parameters. array (N*(N+1),)
       -> p[:N]: growth rates
       -> p[N:]: flattened interaction matrix

    --- OUTPUT ---
    rhs: right-hand-side of gLV. array (M, N,)
    """

    n = x.shape[1]
    r = p[:n]
    A = p[n:].reshape((n, n))
    
    rhs_time = np.zeros((t.shape[0], x.shape[1]))

    for i in range(t.shape[0]):
        rhs_time[i] = x[i] * (r + np.dot(A, x[i]))

    return rhs_time


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
    jac: jacobian matrix of the gLV system for given state x and parameters p. array (N, N*(N-1),)
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
def euler_maruyama(f, t0, x, p, noise_scale, dt, t_eval, seed=0):
    """
    euler_maruyama: Euler-Maruyama method for approximate solution of stochastic differential equations (SDEs)
    N = dimension of the system

    --- INPUT ---
    f: rhs of SDE. function f(t, x, p)
    t: initial time. scalar
    x: initial state of the system. array (N,)
    p: parameters of the model. array (N*(N-1),)
    noise: constant scale multiplying noise. scalar or array (N,)
    dt: size of time step for SDE integration. scalar
    n_steps: number of steps to integrate along. integer
    save_interval: number of dt intervals between sampling the SDE solution. integer

    --- OUTPUT ---
    x_: time-series generated as solution of the SDE. array (len(t_eval), N,)
    """
    
    dt_sqrt = dt**0.5
    n = x.shape[0]
    
    np.random.seed(seed)
    noise = noise_scale*np.random.randn(int((t_eval[-1]-t0)/dt)+len(t_eval), n)

    x_ = np.zeros((t_eval.shape[0], n))

    x_now = x.copy()
    t = t0

    dt_count = 0
    for i in range(len(t_eval)):
        n_steps = (t_eval[i]-t)/dt

        if n_steps == int(n_steps):
            eps = n_steps - int(n_steps)
            n_steps = int(n_steps-1)
        else:
            eps = 0.
            n_steps = int(n_steps)

        for j in range(n_steps):
            x_now += dt * f(t, x_now, p) + dt_sqrt * x_now*noise[dt_count]
            x_now[x_now < 0.] = 0.
            t += dt
            dt_count += 1
        
        x_now += eps*dt * f(t, x_now, p) + eps**0.5*dt_sqrt * x_now*noise[dt_count]
        x_now[x_now < 0.] = 0.
        x_[i] = x_now

        dt_count += 1

    return x_


def sort_glv_params(n, seed, r_max, A_diag_std, A_off_diag_std):
    """
    sort_glv_params: sorts parameters that lead to a stable gLV system
    growth rates are sorted uniformly between 0 and r_max
    interspecies interactions are sorted normally around zero with A_off_diag_std standard deviation
    intraspecies interactions are sorted as negative absolute values of variables normally around zero with A_diag_std standard deviation

    --- INPUT ---
    n: number of species. scalar
    seed: random seed to initialize parameters. integer
    r_max: maximum growth rate. scalar
    A_diag_std: standard deviation for distribution of intraspecies interactions. scalar
    A_off_diag_std: standard deviation for distribution of interspecies interactions. scalar

    --- OUTPUT ---
    p: chosen parameter set. array (N*(N-1),)
    """

    np.random.seed(seed)
    r = np.random.uniform(0, r_max, n)
    A = np.random.normal(scale=A_off_diag_std, size=(n,n))
    for i in range(n):
        A[i,i] = -np.abs(np.random.normal(A_diag_std))

    mat_c = 0
    while (-np.linalg.inv(A)@r < 0).any() or (np.linalg.eig(-np.linalg.inv(A)@r.reshape((-1,1))*A)[0]>0).any():
        mat_c += 1
        print(f"\rnew matrix {mat_c}", end="")
        A = np.random.normal(0, size=(n,n))
        for i in range(n):
            A[i,i] = -np.abs(np.random.normal(3, 0.1))

    p = np.concatenate((r, A.flatten()))

    return p