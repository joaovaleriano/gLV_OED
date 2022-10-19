#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Implementation of inference methods
"""

#%%
# packages

import numpy as np
from numba import njit
import pandas as pd
from stochastic_glv_generator import *
from analysis import *
from sklearn import linear_model as lm

#%%
# linear regression w/ different regularizations

def fit_lr(df, averaging="none"):
    add_log_glv_rhs(df)

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
    add_log_glv_rhs(df)

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
    add_log_glv_rhs(df)

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
def mse_grad_p(rhs, rhs_grad_p, dydt):
    return 2*(np.expand_dims(rhs-dydt, -1)*rhs_grad_p).mean((0, 1))


@njit
def glv_and_jac_time(t, x, p):
    n = x.shape[1]

    x_t = np.zeros((len(t), n))
    x_grad_p_t = np.zeros((len(t), n, n*(n+1)))
    
    for i in range(len(t)):
        x_t[i] = glv(t[i], x[i], p)
        x_grad_p_t[i] = glv_jac(t[i], x[i], p)[1]
        
    return np.array(x_t), np.array(x_grad_p_t)

@njit
def jac_time(t, x, p):
    n = x.shape[1]

    x_grad_p_t = np.zeros((len(t), n, n*(n+1)))
    
    for i in range(len(t)):
        x_grad_p_t[i] = glv_jac(t[i], x[i], p)[1]
        
    return x_grad_p_t


@njit
def mini_batch_sgd_rmsprop(df, p, alpha, gamma, eps, Eg2, batch_size, averaging="none"):
    if averaging == "none":
        y = df.dropna()[[i for i in df.columns if i[:2]=="sp"]]

    elif averaging == "arithm":
        add_arithm_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:14]=="arithm_mean_sp"]]

    elif averaging == "geom":
        add_geom_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:12]=="geom_mean_sp"]]

    t = df.dropna()["time"].values

    add_glv_rhs(df)
    
    dydt = df.dropna()[[i for i in df.columns if i[:6]=="dlogsp"]]

    idxs = list(np.arange(t.shape[0]-1))
    
    np.random.shuffle(idxs)

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