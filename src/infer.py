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

datapath = "../experiment_outputs/perturb_meas_0.1/datasets/"
datafiles = os.listdir(datapath)
metadatafiles = [f"metadata{i.split('dataset')[1].split('csv')[0]}txt"\
                for i in datafiles]

datafiles = natural_sort(datafiles)
metadatafiles = natural_sort(metadatafiles)


#%% set number of species to consider

n_sp = 5
print(f"{n_sp} species")

datafiles_n_sp = [i for i in datafiles if f"{n_sp}_sp" in i]
metadatafiles_n_sp = [i for i in metadatafiles if f"{n_sp}_sp" in i]

df = pd.read_csv(datapath+datafiles[-1], index_col=0)
metatext = open(f"{datapath}../metadata/"+metadatafiles[-1], "r").read().split("\n")


#%% get metadata

metadict = get_meta(metatext)

print(f"Numbers of sampling points: {metadict['n_tpoints']}")
print(f"Average sampling intervals: {metadict['avg_dt'].round(3)}")
print(f"Number of initial conditions: {metadict['n_init_cond']}")
print(f"Number of repetitions: {metadict['repetitions']}")
print(f"Environmental noise: {metadict['env_noise']}")
print(f"Amounts of measurement noise: {metadict['meas_noise']}")

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


#%% Infer and score

try:
    os.mkdir(f"{datapath}/../inference")
except:
    pass

param_columns = [f"r{i}" for i in range(1, n_sp+1)] + \
                [f"A{i},{j}" for i in range(1, n_sp+1) for j in range(1, n_sp+1)]

cols = ["n_init_cond"] + list(df.columns[1:4]) + param_columns + ["MSPD", "CSR", "ES"]
infer_out_all = []

for file_idx in range(len(datafiles_n_sp)):
    datafile = datafiles_n_sp[file_idx]
    metadatafile = metadatafiles_n_sp[file_idx]

    df = pd.read_csv(datapath+datafile, index_col=0)
    metatext = open(f"{datapath}../metadata/"+metadatafile, "r").read().split("\n")
    metadict = get_meta(metatext)
    
    infer_out = pd.DataFrame(columns=cols)

    pd.options.mode.chained_assignment = None

    p = metadict["parameters"]
    r = p[:n_sp]
    A = p[n_sp:].reshape((n_sp,n_sp))

    for t_samp in df.t_samp_dist_idx.unique():
        for meas_noise in df.measurement_noise.unique():
            df_tmp = df[(df[["t_samp_dist_idx", "measurement_noise"]]==[t_samp, meas_noise]).all(axis=1)]
            for i in tqdm(range(len(df.init_cond_idx.unique()))):
                combs = list(combinations(df.init_cond_idx.unique(), i+1))
                np.random.shuffle(combs)
                for comb in combs[:100]:
                    df_comb = df_tmp[df_tmp.init_cond_idx.isin(comb)]
                    r_est, A_est = fit_ridge_cv(df_comb)
                    p_est = np.concatenate((r_est, A_est.flatten()))
                    MSPD = ((p-p_est)**2).mean()
                    CSR = (np.sign(A_est)==np.sign(A)).mean()
                    ES = calculate_es_score(A, A_est)
                    infer_out.loc[len(infer_out)] = [i+1, comb, t_samp, meas_noise] + list(p_est) + [MSPD, CSR, ES]


    infer_out.to_csv(f"{datapath}/../inference/infer_out_"+datafile.split("dataset")[1])
    infer_out_all.append(infer_out)