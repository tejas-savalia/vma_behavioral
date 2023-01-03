# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:20:50 2020

@author: Tejas
"""
#%%

import numpy as np
import scipy.io
from multiprocessing import Pool
from functools import partial
import pickle
import scipy
import scipy.optimize
from sklearn.metrics import *
import scipy.stats as stat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d



#%% Run this to compile dual state model functions

def dual_model_sudden(num_trials, Af, Bf, As, Bs):
    errors = np.zeros((num_trials))
    rotation = 1.0
    fast_est = np.zeros((num_trials))
    slow_est = np.zeros((num_trials))
    rotation_est = np.zeros((num_trials))
    #rotation_est[0] = est
    for trial in range(num_trials - 1):
        if trial < 640:
            rotation = 1.0
            errors[trial] = rotation - rotation_est[trial]
            fast_est[trial+1] = Af*fast_est[trial] + Bf*errors[trial]
            slow_est[trial+1] = As*slow_est[trial] + Bs*errors[trial]
        else:
            rotation = 0
            errors[trial] = rotation_est[trial]
        #print(errors[trial])
            fast_est[trial+1] = Af*fast_est[trial] - Bf*errors[trial]
            slow_est[trial+1] = As*slow_est[trial] - Bs*errors[trial]

        rotation_est[trial+1] = fast_est[trial+1] + slow_est[trial+1]
        #print (rotation_est)
    errors[num_trials-1] = rotation_est[num_trials-1]
    return errors, rotation_est, fast_est, slow_est

def dual_model_gradual(num_trials, Af, Bf, As, Bs):
    errors = np.zeros((num_trials))
    fast_est = np.zeros((num_trials))
    slow_est = np.zeros((num_trials))
    rotation_est = np.zeros((num_trials))
    rotation = 0
    for trial in range(num_trials - 1):
        if trial < 640:
            if trial%64 == 0:
                rotation = rotation + 10/90.0
            if rotation > 1.0:
                rotation = 1.0
            errors[trial] = rotation - rotation_est[trial]
            fast_est[trial+1] = Af*fast_est[trial] + Bf*errors[trial]
            slow_est[trial+1] = As*slow_est[trial] + Bs*errors[trial]
        else:
            rotation = 0
            errors[trial] = rotation_est[trial] 
            fast_est[trial+1] = Af*fast_est[trial] - Bf*errors[trial]
            slow_est[trial+1] = As*slow_est[trial] - Bs*errors[trial]

        rotation_est[trial+1] = fast_est[trial+1] + slow_est[trial+1]
        #print (rotation_est)
    errors[num_trials-1] = rotation_est[num_trials-1]

    return errors, rotation_est, fast_est, slow_est

def residuals_sudden(params, num_trials, data_errors, train_indices):
    model_errors = dual_model_sudden(num_trials, params[0], params[1], params[2], params[3])[0]
    model_errors_train = np.take(model_errors, train_indices)
    data_errors_train = np.take(data_errors, train_indices)
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[4]))
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    if params[0] > params[2]:
        residual_error = residual_error + 10000000
    if params[1] < params[3]:
        residual_error = residual_error + 10000000
    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
        residual_error = residual_error + 10000000
    return residual_error

def residuals_gradual(params, num_trials, data_errors, train_indices):
    model_errors = dual_model_gradual(num_trials, params[0], params[1], params[2], params[3])[0]
    model_errors_train = np.take(model_errors, train_indices)
    data_errors_train = np.take(data_errors, train_indices)
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[4]))
    if params[0] > params[2]:
        residual_error = residual_error + 10000000
    if params[1] < params[3]:
        residual_error = residual_error + 10000000
    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
        residual_error = residual_error + 10000000
    return residual_error



#%% Run this to compile fit routines
    
def fit_participant(participant, curvatures, num_fits):
    Af = np.zeros((num_fits))
    Bf = np.zeros((num_fits))
    As = np.zeros((num_fits))
    Bs = np.zeros((num_fits))
    epsilon = np.zeros((num_fits))
    V = np.zeros((num_fits))
    train_indices = np.zeros((num_fits, 634))
    
    for fit_parts in range(num_fits):
        train_indices = np.random.choice(704, 634, replace = False)
        starting_points = np.array([[0.9, 0.3, 0.99, 0.01, 0.05]])
        for initial_point in starting_points:
            if participant%4 == 0 or participant%4 == 1:      
                fits = scipy.optimize.basinhopping(residuals_sudden, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3], initial_point[4]], minimizer_kwargs={'args': (704, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), 'method':'Nelder-Mead'})

                Af[fit_parts] = fits.x[0]
                Bf[fit_parts] = fits.x[1]
                As[fit_parts] = fits.x[2]
                Bs[fit_parts] = fits.x[3]
                epsilon[fit_parts] = fits.x[4]
                V[fit_parts] = fits.fun#
            else:
                fits = scipy.optimize.basinhopping(residuals_gradual, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3], initial_point[4]], minimizer_kwargs={'args': (704, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), 'method':'Nelder-Mead'})
                Af[fit_parts] = fits.x[0]
                Bf[fit_parts] = fits.x[1]
                As[fit_parts] = fits.x[2]
                Bs[fit_parts] = fits.x[3]
                epsilon[fit_parts] = fits.x[4]
                V[fit_parts] = fits.fun
                
            print (participant, V)
    return Af, Bf, As, Bs, V, epsilon, train_indices

def run_fits_dual(curvatures, num_trials, part_size):
    func = partial(fit_participant, curvatures = curvatures, num_fits = 1)
    pool = Pool()
    res = np.zeros((100), dtype = object)
    for fit in range(100):
        res[fit] = np.reshape(np.array(pool.map(func, range(60))), (60, 7))
        print ('mean fits: ', np.mean(res[fit][:, 4]))
    #return fit_Af, fit_Bf, fit_As, fit_Bs, fit_V
    return res   

def main():
    
    curvatures_smooth = pickle.load(open('curvatures_smooth.pickle', 'rb'))
    curvatures_smooth = curvatures_smooth/90

    print("parallel curvatures successful")
    print (curvatures_smooth)
    
    print ("Curvatures Loaded. In Fit routine")

    #%% Parallel run and dump fits
    fits = run_fits_dual(curvatures_smooth, 640, 640)
    with open('fit_dual_ll_transfer_test_rep.pickle', 'wb') as f:
        pickle.dump(fits, f)
    f.close()
        
    
    
if __name__ == '__main__':
    main()

