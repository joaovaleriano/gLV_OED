#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Direct script to infer parameters from generated datasets with ridge regression (sklearn.lm.RidgeCV).
"""

#%% packages

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pandas as pd
import os
from itertools import combinations
import h5py

import sys
sys.path.append("../src")

from analysis import *
from inference import *

import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)


#%% # set path with data

datapath = "../experiment_outputs/growth_scale_0.1_10_sp_5_param_seed_20_init_cond_extend_env_noise0.0"
log = h5py.File(f"{datapath}/data_generation_log.h5", "r")

print(f"n_species = {log.attrs['n_species']}")
print(f"avg_samp_dt = {log.attrs['avg_samp_dt']}")
print(f"env_noise = {log.attrs['env_noise']}")
print(f"meas_noise_list = {log.attrs['meas_noise_list']}")
print(f"n_params_seeds = {log.attrs['n_params_seeds']}")

env_noise = log.attrs['env_noise']

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


import math

def n_comb(n, k):
    return math.factorial(n)/(math.factorial(n-k)*math.factorial(k))


#%%

# Infer and score

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
                cols = ["n_dset"] + list(df.columns[1:4]) + param_columns + ["MSPD", "CSR", "ES"]

                infer_out = pd.DataFrame(columns=cols)

                pd.options.mode.chained_assignment = None
                
                p = metadict["parameters"]
                r = p[:n_sp]
                A = p[n_sp:].reshape((n_sp,n_sp))

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
                # infer_out.to_csv(datafile.split('dataset')[0]+"/inference_alphas"+datafile.split("dataset")[1])
                # infer_out.to_csv(datafile.split('dataset')[0]+"/inference_lasso"+datafile.split("dataset")[1])
                # infer_out.to_csv(datafile.split('dataset')[0]+"/inference_elasticnet"+datafile.split("dataset")[1])