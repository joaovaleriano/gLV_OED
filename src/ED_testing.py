#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Functions for generating multiple experimental design conditions
"""


#%%
# packages

import numpy as np
from numba import njit

#%%

def subsampler(t, tp, xp):
    """
    subsampling: samples a trajectory xp, orignially measured at time points tp, on points t, by linear interpolation
    
    --- INPUT ---
    t: time points where to sample. array (P,)
    tp: original timepoints. array (N,)
    xp: measured trajectory over points tp. array (N, M)

    --- OUTPUT ---
    x:
    """

    x = np.array([np.interp(t, tp, xp[:,i]) for i in range(xp.shape[1])]).T

    return x

# @njit
def noisy_measurement(x, sig, seed=0):
    """
    noisy_measurement: add gaussian noise of scale sig to data x

    --- INPUT ---
    xn: data do add noise to. array (N, M)
    sig: noise scale / standard deviation of noise's normal distribution. scalar or array (M,)

    --- OUTPUT ---
    xn: data with noise
    """

    np.random.seed(seed)

    xn = x + np.random.normal(scale=sig, size=x.shape)

    return xn