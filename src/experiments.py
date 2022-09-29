#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Generating experiment replicates varying initial conditions, sampling strategy, 
and environmental and measurement noise intensity.
"""


#%%
# packages

import numpy as np
from numba import njit
from stochastic_glv_generator import *

#%%
# functions

def measurement(t, tp, xp, noise=0., seed=0):
    """
    measurement: samples a trajectory xp, orignially measured at time points tp, on points t, by linear interpolation, adding measurement noise
    
    --- INPUT ---
    t: time points where to sample. array (P,)
    tp: original timepoints. array (N,)
    xp: measured trajectory over points tp. array (N, M,)
    nosie: noise scale / standard deviation of noise's normal distribution. scalar or array (M,)

    --- OUTPUT ---
    x: subsampled trajectories with add noise. array (len(t), M,)
    """

    np.random.seed(seed)

    x = np.array([np.interp(t, tp, xp[:,i]) for i in range(xp.shape[1])]).T

    x += np.random.normal(scale=noise, size=x.shape)

    return x

def replicates(x, t, t_samp_list, meas_noise_list, seed):
    """
    replicates: generates multiple replicates of an experiment varying initial conditions,
    sampling strategy, and environmental and measurement noise intensity.
    
    --- INPUT ---
    x0: initial conditions list
    meas_nosie: measurement noise scale / standard deviation of  measurement noise's normal distribution scalar or array (M,)

    --- OUTPUT ---
    
    """

    n = x.shape[1]

    dataset = []

    meas_noise_seeds = np.random.randint(0, np.iinfo(int).max)

    for t_samp in t_samp_list:
        for meas_noise in meas_noise_list:
            dataset.append(np.hstack((t_samp, measurement(t_samp, t, x, meas_noise, seed))))

    return dataset