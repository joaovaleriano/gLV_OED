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
import h5py
import time
import datetime

import sys
sys.path.append("../src")

from stochastic_glv_generator import *
from experiments import *
from analysis import *


#%%
# define dataset properties

# n_sp = np.array([3, 5, 7, 10, 20])
# n_samples = [11, 21, 31, 51, 76, 101, 201]
# t_samp_list = [np.linspace(0, 30, i) for i in n_samples]

# params_seeds = np.arange(100)

# env_noise_list = [0.1]
# meas_noise_list = [0.1]

# n_init_cond = 100

n_sp = np.array([3, 5, 10])
n_samples = [11, 21, 31, 51]
t_samp_list = [np.linspace(0, 30, i) for i in n_samples]

params_seeds = np.arange(10)

env_noise_list = [0.1]
meas_noise_list = [0.1]

n_init_cond = 20

perturb_exp_base = 2.

perturb_scale_avg_and_sig = [0., 1.]

if len(sys.argv) > 1:
    perturb_scale_avg_and_sig[1] = np.float64(sys.argv[1])

save_loc = "test_perturb"
if len(sys.argv) > 2:
    save_loc = sys.argv[2]
print(f"Save location = {save_loc}\n")

repetitions = 1

t0 = 0.
dt = 1e-3

r_max = 1.
A_off_diag_std = 1.
A_diag_mean = A_off_diag_std * n_sp**0.5 * 1.1
A_diag_std = 0.1 * A_diag_mean


#%%

n_experiments = len(n_sp) * len(params_seeds) * len(env_noise_list)
print(f"Number of experiments (= number of files): {n_experiments}")

n_datasets = n_init_cond * repetitions * len(meas_noise_list) * len(n_samples)
print(f"Number of datasets per file: {n_datasets}")

print(f"Total number of datasets: {n_experiments*n_datasets}")

data_size = (n_sp.sum()+5*len(n_sp)) * np.sum(n_samples) *  len(params_seeds) * len(env_noise_list) *\
     len(meas_noise_list) * n_init_cond * repetitions * 18 / 1024**2
print(f"Expected total size: {data_size:.3f} MB")


#%%
# data generation command - perturbation

np.random.seed(0)

for k, env_noise in enumerate(env_noise_list):
    save_loc_k = f"{save_loc}_env_noise{env_noise}"
    
    os.mkdir(f"../experiment_outputs/{save_loc_k}")
    
    for i in range(len(n_sp)):
        os.mkdir(f"../experiment_outputs/{save_loc_k}/{n_sp[i]}_sp")

        print(f"{n_sp[i]} species: ")
        for j in tqdm(range(len(params_seeds))):
            p = sort_glv_params(n_sp[i], params_seeds[j], r_max, A_diag_mean[i], A_diag_std[i], A_off_diag_std)

            r = p[:n_sp[i]]
            A = p[n_sp[i]:].reshape((n_sp[i], n_sp[i]))
            x_eq = -np.linalg.inv(A)@r

            init_cond_list = init_cond_by_perturb(x_eq, perturb_exp_base, perturb_scale_avg_and_sig, n_init_cond)

            save_name = f"{n_sp[i]}_sp{j}_env_noise{env_noise}"

            gen_replicates(p, env_noise, init_cond_list, t0, dt, t_samp_list, meas_noise_list, repetitions, 
            seed=k, scale_meas_noise_by_abund=True, save_datasets=True, save_loc=save_loc_k+f"/{n_sp[i]}_sp/param_seed{j}", save_name=save_name)

    with h5py.File(f"../experiment_outputs/{save_loc_k}/data_generation_log.h5", "w") as log:

        log.attrs["env_noise"] = env_noise
        log.attrs["meas_noise_list"] = meas_noise_list
        log.attrs["n_species"] = n_sp
        log.attrs["n_samples"] = n_samples
        log.attrs["avg_samp_dt"] = [np.diff(t_samp).mean() for t_samp in t_samp_list]
        log.attrs["n_params_seeds"] = len(params_seeds)
        log.attrs["n_init_cond"] = n_init_cond
        log.attrs["repetitions"] = repetitions

        log.attrs["t0"] = t0
        log.attrs["dt"] = dt
        log.attrs["r_max"] = r_max
        log.attrs["A_off_diag_std"] = A_off_diag_std
        log.attrs["A_diag_mean"] = A_diag_mean
        log.attrs["A_diag_std"] = A_diag_std

        log.attrs["perturb_exp_base"] = perturb_exp_base
        log.attrs["perturb_scale_avg_and_sig"] = perturb_scale_avg_and_sig