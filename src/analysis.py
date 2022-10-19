#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Analysis of the dataset to calculate the (log) time derivatives and arithmetic and geometric means for using in regression.
"""

#%%
# packages
import numpy as np
import pandas as pd
from stochastic_glv_generator import glv_time
from sklearn import linear_model as lm

#%%
# manipulations of dataset

def add_time_diff(df):
    species = [i for i in df.columns if i[:2]=="sp"]

    datasets = df["dataset"].unique()

    dspdt_cols = [f"d{i}/dt" for i in species]
    df[dspdt_cols] = np.nan

    for i in datasets:
        dataset = df[df["dataset"]==i]
        idxs = dataset.iloc[:-1].index
        y = dataset[species].values
        t = dataset["time"].values

        dydt = np.diff(y, axis=0)/np.diff(t).reshape((-1,1))

        df.loc[idxs, dspdt_cols] = dydt


def add_log_time_diff(df):
    species = [i for i in df.columns if i[:2]=="sp"]

    datasets = df["dataset"].unique()

    dlogspdt_cols = [f"dlog{i}/dt" for i in species]
    df[dlogspdt_cols] = np.nan

    for i in datasets:
        dataset = df[df["dataset"]==i]
        idxs = dataset.iloc[:-1].index
        y = dataset[species].values
        t = dataset["time"].values

        dydt = np.diff(np.log(y), axis=0)/np.diff(t).reshape((-1,1))

        df.loc[idxs, dlogspdt_cols] = dydt


def add_glv_rhs(df, p):
    species = [i for i in df.columns if i[:2]=="sp"]

    datasets = df["dataset"].unique()

    glv_rhs_cols = [f"glv_rhs_{i}" for i in species]
    df[glv_rhs_cols] = np.nan

    for i in datasets:
        dataset = df[df["dataset"]==i]
        idxs = dataset.iloc[:-1].index
        y = dataset[species].values
        t = dataset["time"].values

        rhs = glv_time(t[:-1], y[:-1], p)

        df.loc[idxs, glv_rhs_cols] = rhs


def add_arithm_mean(df):
    species = [i for i in df.columns if i[:2]=="sp"]
    n_sp = len(species)

    datasets = df["dataset"].unique()

    arithm_mean_cols = [f"arithm_mean_{i}" for i in species]
    df[arithm_mean_cols] = np.nan

    for i in datasets:
        dataset = df[df["dataset"]==i]
        idxs = dataset.iloc[:-1].index
        y = dataset[species].values

        y_arithm = (y[1:]+y[:-1])/2

        df.loc[idxs, arithm_mean_cols] = y_arithm


def add_geom_mean(df):
    species = [i for i in df.columns if i[:2]=="sp"]
    n_sp = len(species)

    datasets = df["dataset"].unique()

    geom_mean_cols = [f"geom_mean_{i}" for i in species]
    df[geom_mean_cols] = np.nan

    for i in datasets:
        dataset = df[df["dataset"]==i]
        idxs = dataset.iloc[:-1].index
        y = dataset[species].values

        y_geom = np.sqrt(y[1:]*y[:-1])

        df.loc[idxs, geom_mean_cols] = y_geom