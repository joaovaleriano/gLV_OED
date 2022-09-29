#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Functions for generating multiple experimental design conditions
"""


#%%
# packages

import numpy as np


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