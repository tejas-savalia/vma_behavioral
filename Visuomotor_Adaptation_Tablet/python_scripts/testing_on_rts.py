# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 13:48:20 2020

@author: Tejas
"""
import numpy as np
import scipy
import scipy.io
import pickle
import scipy.optimize
from dual_model import *
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
#%% Read RT Data and Parameters

def times(data, block):
    initial_time = scipy.io.loadmat('data/data{data}/initial_time/initial_time{block}.mat'.format(block=str(block), data=str(data)))
    movement_time = scipy.io.loadmat('data/data{data}/movement_time/movement_time{block}.mat'.format(block = str(block), data=str(data)))
    #squares = scipy.io.loadmat('data/participants/data{data}/squares/coordinates/squares{block}.mat'.format(block=str(block), data=str(data)))
    #xdiff = (ideal_traj['idealXs'] - traj['x'])
    #ydiff = (ideal_traj['idealYs'] - traj['y'])
    initial_time = initial_time['initial_time'][:, 0]
    movement_time = movement_time['movement_time'][:, 0]
    return initial_time, movement_time
def get_times():
    initial_times = np.zeros((60, 12, 64))
    movement_times = np.zeros((60, 12, 64))
    for participant in range(60):
        for block in range(12):
            it, mt = times(participant+1000, block)
            initial_times[participant, block] = it
            movement_times[participant, block] = mt
    return initial_times, movement_times

its, mts = get_times()
params = pickle.load(open('fit_dual_bound.pickle', 'rb'))
Af, Bf, As, Bs, V = params.T


#%% Correlations
corr_coefs = np.zeros((4, 15, 15))
for test_participant in range(60):
    rt_data = gaussian_filter(np.ravel(its[test_participant][1:-1]), 2)
    z_rt = stats.zscore(rt_data)
    if test_participant%4 == 0:
        for participant in range(0, 60, 4):
            errors_pred = dual_model_sudden(640, Af[participant], Bf[participant], As[participant], Bs[participant])[0]
            z_errors_pred = stats.zscore(errors_pred)
            corr_coefs[0][int(np.floor(test_participant/4))][int(np.floor(participant/4))] = stats.pearsonr(z_errors_pred, z_rt)[0]

    if test_participant%4 == 1:
        for participant in range(1, 60, 4):
            errors_pred = dual_model_sudden(640, Af[participant], Bf[participant], As[participant], Bs[participant])[0]
            z_errors_pred = stats.zscore(errors_pred)
            corr_coefs[1][int(np.floor(test_participant/4))][int(np.floor(participant/4))] = stats.pearsonr(z_errors_pred, z_rt)[0]

    if test_participant%4 == 2:
        for participant in range(2, 60, 4):
            errors_pred = dual_model_gradual(640, Af[participant], Bf[participant], As[participant], Bs[participant])[0]
            z_errors_pred = stats.zscore(errors_pred)
            corr_coefs[2][int(np.floor(test_participant/4))][int(np.floor(participant/4))] = stats.pearsonr(z_errors_pred, z_rt)[0]

    if test_participant%4 == 3:
        for participant in range(3, 60, 4):
            errors_pred = dual_model_gradual(640, Af[participant], Bf[participant], As[participant], Bs[participant])[0]
            z_errors_pred = stats.zscore(errors_pred)
            corr_coefs[3][int(np.floor(test_participant/4))][int(np.floor(participant/4))] = stats.pearsonr(z_errors_pred, z_rt)[0]
        


