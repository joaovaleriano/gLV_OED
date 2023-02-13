#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Direct script to use the implemented methods to generate mock datasets.
"""


#%%
# packages

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm
import pandas as pd
import os
from itertools import combinations
import h5py
import time
import datetime

import sys
sys.path.append("../src")

from stochastic_glv_generator import *
from experiments import *
from analysis import *

import sys
sys.path.append("../src")

from analysis import *
from inference import *

import re


#%%
# define dataset properties

# n_sp = np.array([3, 5, 7, 10, 20])
# n_samples = [11, 21, 31, 51, 76, 101, 201]
# t_samp_list = [np.linspace(0, 30, i) for i in n_samples]

# params_seeds = np.arange(100)

# env_noise_list = [0.1]
# meas_noise_list = [0.1]

# n_init_cond = 100

n_sp = np.array([10])
n_samples = [31]
t_samp_list = [np.linspace(0, 30, i) for i in n_samples]

params_seeds = np.arange(1)

env_noise_list = [0.0, 0.05]
meas_noise_list = [0.0]

n_init_cond = 5
repetitions = 5

growth_scale = [0.1]

if len(sys.argv) > 1:
    growth_scale[0] = np.float64(sys.argv[1])

save_loc = "test_growth"
if len(sys.argv) > 2:
    save_loc = sys.argv[2]
print(f"Save location = {save_loc}\n")

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
# data generation command - growth

np.random.seed(0)

for k, env_noise in enumerate(env_noise_list):
    save_loc_k = f"{save_loc}_env_noise{env_noise}"
    
    os.mkdir(f"../experiment_outputs/{save_loc_k}")

    for i in range(len(n_sp)):
        os.mkdir(f"../experiment_outputs/{save_loc_k}/{n_sp[i]}_sp")

        print(f"{n_sp[i]} species: ")
        for j in tqdm(range(len(params_seeds))):
            p = sort_glv_params(n_sp[i], params_seeds[j], r_max, A_diag_mean[i], A_diag_std[i], A_off_diag_std)
            # p = sort_glv_params_laplace(n_sp[i], params_seeds[j], r_max, A_diag_mean[i], A_diag_std[i], A_off_diag_std)

            r = p[:n_sp[i]]
            A = p[n_sp[i]:].reshape((n_sp[i], n_sp[i]))
            x_eq = -np.linalg.inv(A)@r

            init_cond_list = init_cond_by_growth(x_eq, growth_scale, n_init_cond)

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

        log.attrs["growth_scale"] = growth_scale


#%%

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)


#%% def calculate_es_score

def calculate_es_score(true_aij, inferred_aij) -> float:
    """GRANT'S edited version to calculate ED score

    Calculate the ecological direction (EDₙ) score (n := number of species in ecosystem).

    Parameters
    ===============
    truth: ndarray(axis0=species_names, axis1=species_names), the ecosystem coefficient matrix used to generate data
    inferred: ndarray(axis0=species_names, axis1=species_names), the inferred ecosystem coefficient matrix
    Returns
    ===============
    ES_score: float
    """

    truth = pd.DataFrame(true_aij).copy()
    inferred = pd.DataFrame(inferred_aij).copy()

    # consider inferred coefficients
    mask = inferred != 0

    # compare sign: agreement when == -2 or +2, disagreement when 0
    nonzero_sign = np.sign(inferred)[mask] + np.sign(truth)[mask]
    corr_sign = (np.abs(nonzero_sign) == 2).sum().sum()
    opposite_sign = (np.abs(nonzero_sign) == 0).sum().sum()

    # count incorrect non-zero coefficients
    wrong_nz = (truth[mask] == 0).sum().sum()

    # combine
    unscaled_score = corr_sign - opposite_sign

    # scale by theoretical extrema
    truth_nz_counts = (truth != 0).sum().sum()
    truth_z_counts = len(truth.index) ** 2 - truth_nz_counts
    theoretical_min = -truth_nz_counts
    theoretical_max = truth_nz_counts

    ES_score = (unscaled_score - theoretical_min) / (theoretical_max - theoretical_min)

    return ES_score


#%%

def get_files(datapath, n_sp, env_noise, meas_noise, avg_samp_dt, filetype="dataset", ext="csv"):
    params_seeds = [i.split("param_seed")[1] for i in os.listdir(f"{datapath}/{n_sp}_sp")]

    datafiles = []

    for p in params_seeds:
        datafiles.append(f"{datapath}/{n_sp}_sp/param_seed{p}/meas_noise{meas_noise}/t_samp{avg_samp_dt}/{filetype}{n_sp}_sp{p}_env_noise{env_noise}.{ext}")
    return datafiles


def n_comb(n, k):
    return math.factorial(n)/(math.factorial(n-k)*math.factorial(k))


#%%

# Infer and score

# for env_noise_k in env_noise_list:
#     datapath = f"../experiment_outputs/{save_loc}_env_noise{env_noise_k}"
#     log = h5py.File(f"{datapath}/data_generation_log.h5", "r")

#     print(f"n_species = {log.attrs['n_species']}")
#     print(f"avg_samp_dt = {log.attrs['avg_samp_dt']}")
#     print(f"env_noise = {log.attrs['env_noise']}")
#     print(f"meas_noise_list = {log.attrs['meas_noise_list']}")
#     print(f"n_params_seeds = {log.attrs['n_params_seeds']}")

#     env_noise = log.attrs['env_noise']

#     for n_sp in log.attrs["n_species"]:
#         for avg_samp_dt in log.attrs["avg_samp_dt"]:
#             for meas_noise in log.attrs["meas_noise_list"]:
#                 datafiles = get_files(datapath, n_sp, env_noise, meas_noise, avg_samp_dt)
#                 metadatafiles = get_files(datapath, n_sp, env_noise, meas_noise, avg_samp_dt, "metadata", "txt")

#                 for file_idx in range(len(datafiles)):
#                     datafile = datafiles[file_idx]
#                     metadatafile = metadatafiles[file_idx]
#                     metadict = get_meta(open(metadatafile, "r").read().split("\n"))
                    
#                     df = pd.read_csv(datafile, index_col=0)
                    
#                     param_columns = [f"r{i}" for i in range(1, n_sp+1)] + \
#                     [f"A{i},{j}" for i in range(1, n_sp+1) for j in range(1, n_sp+1)]
#                     cols = ["n_dset"] + list(df.columns[1:5]) + param_columns + ["MSPD", "CSR", "ES"]

#                     infer_out = pd.DataFrame(columns=cols)

#                     pd.options.mode.chained_assignment = None
                    
#                     p = metadict["parameters"]
#                     r = p[:n_sp]
#                     A = p[n_sp:].reshape((n_sp,n_sp))

#                     for i in tqdm(range(len(df.dataset.unique()))):
#                     # for i in tqdm(range(30)):
#                         if n_comb(len(df.dataset.unique()), i+1) < 10000:
#                             combs = list(combinations(df.dataset.unique(), i+1))
#                             np.random.shuffle(combs)
#                             combs = combs[:100]
#                         else:
#                             combs = []
#                             while len(combs) < 100:
#                                 comb = tuple(np.random.choice(df.dataset.unique(), i+1, replace=False))
#                                 if comb not in combs:
#                                     combs.append(comb)
#                         for comb in combs:
#                             comb = np.random.choice(df.dataset.unique(), i+1, replace=False)
#                             df_comb = df[df.dataset.isin(comb)]
#                             r_est, A_est = fit_ridge_cv(df_comb)
#                             # r_est, A_est = fit_lasso_cv(df_comb)
#                             # r_est, A_est = fit_elasticnet_cv(df_comb)
#                             p_est = np.concatenate((r_est, A_est.flatten()))
#                             MSPD = ((p-p_est)**2).mean()
#                             CSR = (np.sign(A_est)==np.sign(A)).mean()
#                             ES = calculate_es_score(A, A_est)
#                             infer_out.loc[len(infer_out)] = [i+1, comb, avg_samp_dt, meas_noise] + list(p_est) + [MSPD, CSR, ES]

#                     infer_out.to_csv(datafile.split('dataset')[0]+"/inference"+datafile.split("dataset")[1])

#%%

# Infer and score

for env_noise_k in env_noise_list:
    datapath = f"../experiment_outputs/{save_loc}_env_noise{env_noise_k}"
    log = h5py.File(f"{datapath}/data_generation_log.h5", "r")

    print(f"n_species = {log.attrs['n_species']}")
    print(f"avg_samp_dt = {log.attrs['avg_samp_dt']}")
    print(f"env_noise = {log.attrs['env_noise']}")
    print(f"meas_noise_list = {log.attrs['meas_noise_list']}")
    print(f"n_params_seeds = {log.attrs['n_params_seeds']}")

    env_noise = log.attrs['env_noise']

    for n_sp in log.attrs["n_species"]:
        for avg_samp_dt in log.attrs["avg_samp_dt"]:
            for meas_noise in log.attrs["meas_noise_list"]:
                datafiles = get_files(datapath, n_sp, env_noise, meas_noise, avg_samp_dt)
                metadatafiles = get_files(datapath, n_sp, env_noise, meas_noise, avg_samp_dt, "metadata", "txt")

                for file_idx in range(len(datafiles)):
                    datafile = datafiles[file_idx]
                    metadatafile = metadatafiles[file_idx]
                    metadict = get_meta(open(metadatafile, "r").read().split("\n"))
                    
                    df = pd.read_csv(datafile, index_col=0)
                    
                    param_columns = [f"r{i}" for i in range(1, n_sp+1)] + \
                    [f"A{i},{j}" for i in range(1, n_sp+1) for j in range(1, n_sp+1)]
                    cols = ["n_dset"] + list(df.columns[1:5]) + param_columns + ["MSPD", "CSR", "ES"]

                    infer_out = pd.DataFrame(columns=cols)

                    pd.options.mode.chained_assignment = None
                    
                    p = metadict["parameters"]
                    r = p[:n_sp]
                    A = p[n_sp:].reshape((n_sp,n_sp))

                    for init_cond_idx in df.init_cond_idx.unique():
                        df_init_cond = df[df["init_cond_idx"]==init_cond_idx]
                        for i in tqdm(range(len(df.dataset.unique()))):
                        # for i in tqdm(range(30)):
                            if n_comb(len(df.dataset.unique()), i+1) < 10000:
                                combs = list(combinations(df.dataset.unique(), i+1))
                                np.random.shuffle(combs)
                                combs = combs[:100]
                            else:
                                combs = []
                                while len(combs) < 100:
                                    comb = tuple(np.random.choice(df.dataset.unique(), i+1, replace=False))
                                    if comb not in combs:
                                        combs.append(comb)
                            for comb in combs:
                                comb = np.random.choice(df.dataset.unique(), i+1, replace=False)
                                df_comb = df[df.dataset.isin(comb)]
                                r_est, A_est = fit_ridge_cv(df_comb)
                                # r_est, A_est = fit_lasso_cv(df_comb)
                                # r_est, A_est = fit_elasticnet_cv(df_comb)
                                p_est = np.concatenate((r_est, A_est.flatten()))
                                MSPD = ((p-p_est)**2).mean()
                                CSR = (np.sign(A_est)==np.sign(A)).mean()
                                ES = calculate_es_score(A, A_est)
                                infer_out.loc[len(infer_out)] = [i+1, comb, avg_samp_dt, meas_noise] + list(p_est) + [MSPD, CSR, ES]

                    infer_out.to_csv(datafile.split('dataset')[0]+"/inference"+datafile.split("dataset")[1])