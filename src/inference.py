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
    """
    fit_lr: fits via sklearn.lm.LinearRegression 
            dependent variable: log(Δy/Δt), regressor: y, 
            y = species abundances, t = time

    --- INPUT ---
    df: pandas DataFrame with data to be fit
    averaging: how to take averages of y to use as regressor. string

    --- OUTPUT ---
    reg.intercept_: obtained growth rates
    reg.coef_: obtained interaction matrix
    """

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
    """
    fit_ridge_cv: fit via sklearn.lm.RidgeCV
            dependent variable: log(Δy/Δt), regressor: y, 
            y = species abundances, t = time

    --- INPUT ---
    df: pandas DataFrame with data to be fit
    averaging: how to take averages of y to use as regressor. string

    --- OUTPUT ---
    reg.intercept_: obtained growth rates
    reg.coef_: obtained interaction matrix
    """

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
    """
    fit_elasticnet_cv: fit via sklearn.lm.MultiTaskElasticNetcv
            dependent variable: log(Δy/Δt), regressor: y, 
            y = species abundances, t = time

    --- INPUT ---
    df: pandas DataFrame with data to be fit
    averaging: how to take averages of y to use as regressor. string

    --- OUTPUT ---
    reg.intercept_: obtained growth rates
    reg.coef_: obtained interaction matrix
    """
    
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
def mse(a, b):

    return ((a-b)**2).mean()


@njit
def mse_grad_p(rhs, rhs_grad_p, dydt):
    """
    mse_grad_p: gradient of MSE between dydt estimator and calculated gLV rhs,
                with respect to parameters

    --- INPUT ---
    rhs:
    rhs_grad_p:
    dydt:

    --- OUTPUT ---
    

    """

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


#%%

def get_meta(metatext):
    """
    
    """

    meta = {}
    for n, line in enumerate(metatext):
        if "initial conditions" in line:
            init_cond_ln_idx = n

        elif "sampling timepoints" in line:
            t_samp_ln_idx = n

        elif "parameters" in line:
            meta["parameters"] = np.array([np.float64(j) for j in line.split(": ")[1].split(",")])
            params_ln_idx = n
        
        elif "measurement noise" in line:
            meta["meas_noise"] = np.array([np.float64(j) for j in line.split(": ")[1].split(",")])
            meas_noise_ln_idx = n
            break

    meta["init_cond"] = np.array([[np.float64(i) for i in metatext[j].split(",")] \
                                  for j in range(init_cond_ln_idx+1, t_samp_ln_idx)])

    meta["n_init_cond"] = len(meta["init_cond"])

    meta["t_samp"] = [np.array([np.float64(i) for i in metatext[j].split(",")]) \
                                  for j in range(t_samp_ln_idx+1, params_ln_idx)]

    meta["n_tpoints"] = np.array([len(t) for t in meta["t_samp"]])

    meta["avg_dt"] = np.array([np.diff(t).mean() for t in meta["t_samp"]])

    for i in range(meas_noise_ln_idx+1, len(metatext)-1):
        key, val = metatext[i].split(": ")
        meta[key] = np.float64(val)

    key, val = metatext[-1].split(": ")
    meta[key] = bool(val)

    meta["repetitions"] = int(meta["repetitions"])

    return meta