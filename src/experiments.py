#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Generating experiment replicates varying initial conditions, sampling strategy, and environmental and measurement noise intensity.
"""


#%%
# packages

import numpy as np
from numba import njit
import pandas as pd
from datetime import datetime
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
    noise: noise scale / standard deviation of noise's normal distribution. scalar or array (M,)

    --- OUTPUT ---
    x: subsampled trajectories with add noise. array (len(t), M,)
    """

    np.random.seed(seed)

    x = np.array([np.interp(t, tp, xp[:,i]) for i in range(xp.shape[1])]).T

    x += np.random.normal(scale=noise, size=x.shape)

    return x


def gen_experiment(p, init_perturb, t0, dt, t_samp, env_noise, meas_noise, perturb_seed=0, env_seed=0, meas_seed=0, scale_meas_noise_by_abund=True):
    """
    gen_experiments:

    --- INPUT ---


    --- OUTPUT ---

    """

    n = int((np.sqrt(1+4*len(p))-1)/2)

    r = p[:n]
    A = p[n:].reshape((n, n))
    x_eq = -np.linalg.inv(A)@r

    np.random.seed(perturb_seed)
    x0 = x_eq + np.random.normal(init_perturb*x_eq, init_perturb*x_eq/10)

    system = euler_maruyama(glv, t0, x0, p, env_noise, dt, t_samp, env_seed)

    if scale_meas_noise_by_abund:
        scaled_meas_noise = meas_noise*data.mean(axis=0)
    else:
        scaled_meas_noise = meas_noise

    data = measurement(t_samp, t_samp, system, scaled_meas_noise, meas_seed)

    return data


def gen_replicates(p, env_noise, init_perturb_list, t0, dt, t_samp_list, meas_noise_list, repetitions, seed=0, scale_meas_noise_by_abund=True, save_datasets=False, save_loc=""):
    """
    replicates: generates multiple replicates of an experiment varying initial conditions,
    sampling strategy, and environmental and measurement noise intensity.
    
    --- INPUT ---
    x: generated data to be measured over multiple replicates. array (N, M,), N = time length, M = number of species
    t_samp_list: list of sampling strategies. list, each entry being an arryay of timepoints where to sample data x
    meas_noise_list: measurement noise scale / standard deviation of  measurement noise's normal distribution. scalar or array (M,)
    repetitions: number of repetitions to do for each set of conditions
    seed: seed to generate seeds of meas. integer
    save_dataset: whether to save final replicates dataset. bool
    save_loc: location to save dataset. string
    dataset_name: name to give the dataset if to be saved. string

    --- OUTPUT ---
    dataframe: pandas DataFrame containing time-series of all generated replicates. pandas.DataFrame
    """

    n = int((np.sqrt(1+4*len(p))-1)/2)

    datasets = []
    
    n_replicates = len(t_samp_list)*len(meas_noise_list)*len(init_perturb_list)*repetitions

    np.random.seed(seed)
    perturb_seeds = np.random.randint(0, 10**9, n_replicates)
    env_seeds = np.random.randint(0, 10**9, n_replicates)
    meas_seeds = np.random.randint(0, 10**9, n_replicates)

    repl_c = 0

    for init_perturb in init_perturb_list:
        for t_samp in t_samp_list:
            for meas_noise in meas_noise_list:            
                for rep in range(repetitions):
                    
                    data = gen_experiment(p, init_perturb, t0, dt, t_samp, env_noise, meas_noise, perturb_seeds[repl_c], env_seeds[repl_c], meas_seeds[repl_c], True)

                    datasets.append(np.hstack((np.array([repl_c, init_perturb, meas_noise])*np.ones((t_samp.shape[0], 3), dtype=int), t_samp.reshape((-1, 1)), data)))

                    repl_c += 1

                    print("\r"+" "*100, end="")
                    print("\r" + f"{repl_c+1}/{n_replicates}", end="")

    datasets = np.vstack(datasets)

    cols = ["dataset", "initial_perturbation", "measurement_noise", "time"] + [f"sp{i}" for i in range(1, n+1)]

    dataframe = pd.DataFrame(data=datasets, columns=cols)

    if save_datasets:
        datetime_now = str(datetime.now()).split(".")[0].replace("-", "").replace(":", "").replace(" ", "-")
        dataframe.to_csv(f"{save_loc}/dataset{datetime_now}.csv")

        # metadata_file = open(f"{save_loc}/metadata{datetime_now}.txt", "w")
        # metadata = [f""]

    return dataframe