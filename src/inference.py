#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Implementation of inference methods
"""

#%%
# packages

import numpy as np
from numba import jit, njit
import pandas as pd
from stochastic_glv_generator import *
from analysis import *
from sklearn import linear_model as lm
from scipy.optimize import curve_fit
from tqdm import tqdm

#%%
# linear regression w/ different regularizations

def fit_lr(df, averaging="none"):
    add_log_time_diff(df)

    reg = lm.LinearRegression()

    if averaging == "none":
        y = df.dropna()[[i for i in df.columns if i[:2]=="sp"]].values

    elif averaging == "arithm":
        add_arithm_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:14]=="arithm_mean_sp"]].values

    elif averaging == "geom":
        add_geom_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:12]=="geom_mean_sp"]].values

    dlogydt = df.dropna()[[i for i in df.columns if i[:6]=="dlogsp"]].values

    reg.fit(y, dlogydt)

    return reg.intercept_, reg.coef_


def fit_ridge_cv(df, averaging="none"):
    add_log_time_diff(df)

    reg = lm.RidgeCV(alphas=10.**np.arange(-5, 3))

    if averaging == "none":
        y = df.dropna()[[i for i in df.columns if i[:2]=="sp"]].values

    elif averaging == "arithm":
        add_arithm_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:14]=="arithm_mean_sp"]].values

    elif averaging == "geom":
        add_geom_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:12]=="geom_mean_sp"]].values

    dlogydt = df.dropna()[[i for i in df.columns if i[:6]=="dlogsp"]].values

    reg.fit(y, dlogydt)

    return reg.intercept_, reg.coef_


def fit_elasticnet_cv(df, averaging="none"):
    add_log_time_diff(df)

    reg = lm.MultiTaskElasticNetCV(alphas=10.**np.arange(-5, 3))

    if averaging == "none":
        y = df.dropna()[[i for i in df.columns if i[:2]=="sp"]].values

    elif averaging == "arithm":
        add_arithm_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:14]=="arithm_mean_sp"]].values

    elif averaging == "geom":
        add_geom_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:12]=="geom_mean_sp"]].values

    dlogydt = df.dropna()[[i for i in df.columns if i[:6]=="dlogsp"]].values

    reg.fit(y, dlogydt)

    return reg.intercept_, reg.coef_


#%%
# gradient descent optimization

@njit
def mse(rhs, dydt):
    return ((rhs-dydt)**2).mean()


@njit
def mse_grad_p(rhs, rhs_grad_p, dydt):
    a = 2*(np.expand_dims(rhs-dydt, -1)*rhs_grad_p)

    r = np.zeros(rhs_grad_p.shape[-1])

    for i in range(rhs.shape[0]):
        for j in range(rhs.shape[1]):
            r += a[i,j]

    return r/(rhs.shape[0]*rhs.shape[1])


@njit
def glv_and_jac_time(t, x, p):
    n = x.shape[1]

    x_t = np.zeros((len(t), n))
    x_grad_p_t = np.zeros((len(t), n, n*(n+1)))
    
    for i in range(len(t)):
        x_t[i] = glv(t[i], x[i], p)
        x_grad_p_t[i] = glv_jac(t[i], x[i], p)
        
    return x_t, x_grad_p_t


@njit
def glv_jac_time(t, x, p):
    n = x.shape[1]

    x_grad_p_t = np.zeros((len(t), n, n*(n+1)))
    
    for i in range(len(t)):
        x_grad_p_t[i] = glv_jac(t[i], x[i], p)
        
    return x_grad_p_t


def mini_batch_sgd_rmsprop(df, p_, alpha, gamma, eps, Eg2, batch_size, averaging="none"):
    p = np.copy(p_)

    if averaging == "none":
        y = df.dropna()[[i for i in df.columns if i[:2]=="sp"]].values

    elif averaging == "arithm":
        add_arithm_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:14]=="arithm_mean_sp"]].values

    elif averaging == "geom":
        add_geom_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:12]=="geom_mean_sp"]].values

    t = df.dropna()["time"].values
    
    dydt = df.dropna()[[i for i in df.columns if i[:6]=="dlogsp"]].values

    idxs = list(np.arange(t.shape[0]-1))

    np.random.shuffle(idxs)

    if batch_size > len(idxs):
        batch_size = len(idxs)

    n_batches = len(idxs)/batch_size
    if np.ceil(n_batches) - n_batches == 0:
        batches = [idxs[batch_size*i:batch_size*(i+1)] for i in range(int(n_batches))]
    
    else:
        batches = [idxs[batch_size*i:batch_size*(i+1)] for i in range(int(np.floor(n_batches)))]
        batches[-1] += [idxs[batch_size*int(np.floor(n_batches))]]
    
    for batch in batches:
        g = mse_grad_p(*glv_and_jac_time(t[batch], y[batch], p), dydt[batch])
        Eg2 = gamma*Eg2 + (1-gamma)*g*g
        p -= alpha/(Eg2+eps)**0.5*g
    
    return p, Eg2


#%%
# Levenberg-Marquardt gradient matching

def glv_for_fit(x, *p):
    n = int((np.sqrt(1+4*len(p))-1)/2)

    x_= x.copy()

    r = np.array(p[:n])
    A = np.array(p[n:]).reshape((n, n))

    rhs = x_*(r + np.dot(x_, A.T))

    return rhs.flatten()


def lm_fit(f, x, y, p0_list, sig=None, maxfev=400):
    p_list = np.zeros_like(p0_list)

    for i in tqdm(range(p0_list.shape[0])):
        p_list[i] = curve_fit(f, x, y, p0_list[i], sigma=sig, maxfev=maxfev)[0]

    return p_list

