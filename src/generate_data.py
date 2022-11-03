#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Direct script to use the implemented methods to generate mock datasets.
"""


#%%
# packages

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime

import sys
sys.path.append("../src")

from stochastic_glv_generator import *
from experiments import *
from analysis import *


#%%
# define dataset properties

n_sp = np.array([2, 3, 4, 5])
n_samples = [5, 10, 20, 50, 100]
t_samp_list = [np.linspace(0, 30, i) for i in n_samples]

params_seeds = np.arange(100)

env_noise_list = [0., 0.05, 0.1, 0.2]
meas_noise_list = [0., 0.05, 0.1, 0.2]

n_init_cond = 5
repetitions = 1

t0 = 0.
dt = 1e-3

r_max = 1.
A_off_diag_std = 1.
A_diag_mean = A_off_diag_std * n_sp**0.5 * 1.1
A_diag_std = 0.1 * A_diag_mean

data_size = n_sp.sum() * np.sum(n_samples) * len(params_seeds) * len(env_noise_list) * len(meas_noise_list) * n_init_cond * repetitions * 8 / 1024**2
print(f"Expected total dataset size: {data_size:.3f} MB")


#%%
# data generation command

for i in range(len(n_sp)):
    print(f"{n_sp[i]} species: ")
    for j in tqdm(range(len(params_seeds))):
        p = sort_glv_params(n_sp[i], params_seeds[j], r_max, A_diag_mean[i], A_diag_std[i], A_off_diag_std)

        r = p[:n_sp[i]]
        A = p[n_sp[i]:].reshape((n_sp[i], n_sp[i]))
        x_eq = -np.linalg.inv(A)@r

        init_cond_list = init_cond_by_perturb(x_eq, 2, [[-1, 1]], 1)

        for env_noise in env_noise_list:
            save_name = f"{n_sp[i]}_sp{j}_env_noise{env_noise}"
            gen_replicates(p, env_noise, init_cond_list, t0, dt, t_samp_list, meas_noise_list, repetitions, seed=0, scale_meas_noise_by_abund=True, save_datasets=True, save_loc="test", save_name=save_name)

