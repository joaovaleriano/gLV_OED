#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: joaovaleriano

Implementation of inference methods
"""

#%%
# packages
import numpy as np
import pandas as pd
from analysis import *
from sklearn import linear_model as lm

#%%
# linear regression w/ different regularizations

def fit_lr(df, averaging="none"):
    add_log_glv_rhs(df)

    reg = lm.LinearRegression()

    if averaging == "none":
        y = df.dropna()[[i for i in df.columns if i[:2]=="sp"]].values

    elif averaging == "arithm":
        add_arithm_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:14]=="arithm_mean_sp"]].values

    elif averaging == "geom":
        add_geom_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:12]=="geom_mean_sp"]].values

    dlogydt = df.dropna()[[i for i in df.columns if i[:6]=="dlogsp"]].values

    reg.fit(y, dlogydt)

    return reg.intercept_, reg.coef_


def fit_ridge_cv(df, averaging="none"):
    add_log_glv_rhs(df)

    reg = lm.RidgeCV(alphas=10.**np.arange(-5, 3))

    if averaging == "none":
        y = df.dropna()[[i for i in df.columns if i[:2]=="sp"]].values

    elif averaging == "arithm":
        add_arithm_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:14]=="arithm_mean_sp"]].values

    elif averaging == "geom":
        add_geom_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:12]=="geom_mean_sp"]].values

    dlogydt = df.dropna()[[i for i in df.columns if i[:6]=="dlogsp"]].values

    reg.fit(y, dlogydt)

    return reg.intercept_, reg.coef_


def fit_elasticnet_cv(df, averaging="none"):
    add_log_glv_rhs(df)

    reg = lm.MultiTaskElasticNetCV(alphas=10.**np.arange(-5, 3))

    if averaging == "none":
        y = df.dropna()[[i for i in df.columns if i[:2]=="sp"]].values

    elif averaging == "arithm":
        add_arithm_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:14]=="arithm_mean_sp"]].values

    elif averaging == "geom":
        add_geom_mean(df)
        y = df.dropna()[[i for i in df.columns if i[:12]=="geom_mean_sp"]].values

    dlogydt = df.dropna()[[i for i in df.columns if i[:6]=="dlogsp"]].values

    reg.fit(y, dlogydt)

    return reg.intercept_, reg.coef_


#%%
# gradient descent optimization

