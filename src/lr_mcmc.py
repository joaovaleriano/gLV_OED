#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 15:00:06 2023

@author: valeriano
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numba import njit


#%%

@njit
def f(x, p):
    return p[0] + (x*p[1:]).sum(1)


@njit
def log_ll_lr(x, y, p_sig):
    p = p_sig[:-1]
    sig = p_sig[-1]
    # p = p_sig.copy()
    # sig = (y-f(x, p)).std()
    # sig = 1e-2
    n_s = x.shape[0]
    
    # if sig <= 0.:
    #     return -np.inf
    
    return -0.5*n_s*np.log(2*np.pi*sig**2) - ((y-f(x, p))**2).sum()/(2*sig**2)


@njit
def multivariate_normal(mean, cov):
    # print(cov)
    x = np.random.randn(len(mean))
    L = np.linalg.cholesky(cov)
    
    return L.dot(x).T + mean


@njit
def tensordot(x, y):
    tp = np.zeros((len(x), len(y)))
    
    for i in range(len(x)):
        tp[i] = x[i]*y
        
    return tp


# @njit
def mcmcMH(log_ll_f, feat, label, params0, prop_sig, cov_burn, cov_burn_rep, n_burn, n_samp):
    samples = np.zeros((n_samp, len(params0)))
    log_ll = np.zeros(n_samp)
    
    params = params0.copy()
    
    m = params0.copy()
    
    cov = np.zeros((params0.shape[0], params0.shape[0]))
    
    for i in range(cov_burn):
        new_params = params + np.random.randn(len(params))*prop_sig**0.5#*1e-1
        # new_params = params + multivariate_normal(np.zeros_like(params), prop_sig*cov)
        
        ln_r = np.log(np.random.rand())
        
        if ln_r < log_ll_f(feat, label, new_params) - log_ll_f(feat, label, params):
            params = new_params.copy()
    
        m_ = m + (params-m)/(i+1)
        cov = (i*cov + tensordot(params, params) - (i+1)*tensordot(m_, m_) + i*tensordot(m, m))/(i+1)
        m = m_.copy()
    
    for k in range(1, cov_burn_rep):
        cov0 = cov.copy()
        cov *= 0
        for i in range(cov_burn):
            # new_params = params + np.random.randn(len(params))*prop_sig**0.5#*1e-1
            new_params = params + multivariate_normal(np.zeros_like(params), prop_sig*cov0)
            
            ln_r = np.log(np.random.rand())
            
            if ln_r < log_ll_f(feat, label, new_params) - log_ll_f(feat, label, params):
                params = new_params.copy()
        
            m_ = m + (params-m)/(i+1)
            cov = (i*cov + tensordot(params, params) - (i+1)*tensordot(m_, m_) + i*tensordot(m, m))/(i+1)
            m = m_.copy()

    for i in range(n_burn):
        # new_params = params + np.random.randn(len(params))*prop_sig**0.5
        new_params = params + multivariate_normal(np.zeros_like(params), prop_sig*cov)
        
        ln_r = np.log(np.random.rand())
        
        if ln_r < log_ll_f(feat, label, new_params) - log_ll_f(feat, label, params):
            params = new_params.copy()
            
        j = cov_burn+1+i
        # j = i
            
        m_ = m + (params-m)/(j+1)
        cov = (j*cov + tensordot(params, params) - (j+1)*tensordot(m_, m_) + \
                + j*tensordot(m, m))/(j+1)
        m = m_.copy()
            
    for i in range(n_samp):
        # new_params = params + np.random.randn(len(params))*prop_sig**0.5
        new_params = params + multivariate_normal(np.zeros_like(params), prop_sig*cov)
        
        ln_r = np.log(np.random.rand())
        
        if ln_r < log_ll_f(feat, label, new_params) - log_ll_f(feat, label, params):
            params = new_params.copy()
        
        j = cov_burn+n_burn+2+i
        # j = n_burn+1+i
            
        m_ = m + (params-m)/(j+1)
        cov = (j*cov + tensordot(params, params) - (j+1)*tensordot(m_, m_) + \
                + j*tensordot(m, m))/(j+1)
        m = m_.copy()

        samples[i] = params
        log_ll[i] = log_ll_f(feat, label, params)
        
    # print(np.diag(cov), np.diag(np.cov(samples.T)))
    
    return samples, log_ll


#%%

np.random.seed(0)

n_s = 100
n_p = 5
x = np.random.rand(n_s, n_p)
interc = np.random.randn()
slope = np.random.randn(n_p)
p = np.concatenate(([interc], slope))

y = f(x, p) + np.random.randn(x.shape[0])*0.01

# plt.subplots(1, n_p, figsize=(20,5))
# for i in range(n_p):
#     plt.subplot(1, n_p, i+1)
#     plt.scatter(x[:,i], y)
# plt.show()

p0 = np.random.randn(n_p+1)
# p0 = p.copy()
# sig0 = np.random.rand()
sig0 = 1.

p_sig0 = np.concatenate((p0, [sig0]))

prop_sig = 2.38**2/(n_p+2)#*1e-2

print(p)

samples, log_ll_hist = mcmcMH(log_ll_lr, x, y, p_sig0, prop_sig, 10000, 5, 100000, 50000)
print(samples.mean(0))
sns.pairplot(pd.DataFrame(data=samples), kind="hist")

# prop_sig *= 1e-1

# samples, log_ll_hist = mcmcMH(log_ll_lr, x, y, p_sig0, prop_sig, 1000, 10, 100000, 50000)
# print(samples.mean(0))
# sns.pairplot(pd.DataFrame(data=samples), kind="hist")