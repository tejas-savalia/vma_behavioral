# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:20:50 2020

@author: Tejas
"""

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

def model_sudden(num_trials, A, B):
    errors = np.zeros((num_trials))
    rotation = 90/90.0
    rotation_est = np.zeros((num_trials))
    for trial in range(num_trials - 1):
        if trial < 640:
            rotation = 90/90.0
            errors[trial] = rotation - rotation_est[trial]
            rotation_est[trial+1] = A*rotation_est[trial] + B*errors[trial]
        else:
            rotation = 0
            errors[trial] = rotation_est[trial]
            rotation_est[trial+1] = A*rotation_est[trial] - B*errors[trial]
        #errors[trial] = rotation - rotation_est[trial]
    errors[num_trials-1] = rotation_est[num_trials-1]
    return errors, rotation_est

def model_gradual(num_trials, A, B):
    errors = np.zeros((num_trials))
    rotation_est = np.zeros((num_trials))
    rotation = 0
    for trial in range(num_trials - 1):
        if trial < 640:
            if trial%64 == 0:
                rotation = rotation + 10/90.0
            if rotation > 1.0:
                rotation = 1.0
            errors[trial] = rotation - rotation_est[trial]
            rotation_est[trial+1] = A*rotation_est[trial] + B*errors[trial]
        else:
            rotation = 0
            errors[trial] = rotation_est[trial]
            rotation_est[trial+1] = A*rotation_est[trial] - B*errors[trial]

    errors[num_trials-1] = rotation_est[num_trials-1]
    return errors, rotation_est

def residuals_sudden(params, num_trials, data_errors):
    model_errors = model_sudden(num_trials, params[0], params[1])[0]
    #model_errors_train = np.take(model_errors, train_indices)
    #data_errors_train = np.take(data_errors, train_indices)
    #residual_error = np.sum((model_errors - data_errors)**2)
    exp_quantiles = np.quantile(model_errors, [0.1, 0.3, 0.5, 0.7, 0.9])
    exp_bin_counts = num_trials*np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    q_counts = list()
    for i in exp_quantiles:
        q_counts.append(sum(data_errors < i))
    q_counts.insert(0, 0)
    q_counts.append(640)
    obs_bin_counts = np.diff(q_counts)
    log_val = np.log(obs_bin_counts/exp_bin_counts)
    log_val[np.isneginf(log_val)] = 0 

    residual_error = 2*sum(np.array(obs_bin_counts)*log_val) 

    if params[0] < 0 or params[1] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1:
        residual_error = residual_error + 10000000
    return residual_error

def residuals_gradual(params, num_trials, data_errors):
    model_errors = model_gradual(num_trials, params[0], params[1])[0]
    #model_errors_train = np.take(model_errors, train_indices)
    #data_errors_train = np.take(data_errors, train_indices)
    #residual_error = np.sum((model_errors - data_errors)**2)
    exp_quantiles = np.quantile(model_errors, [0.1, 0.3, 0.5, 0.7, 0.9])
    exp_bin_counts = num_trials*np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    q_counts = list()
    for i in exp_quantiles:
        q_counts.append(sum(data_errors < i))
    q_counts.insert(0, 0)
    q_counts.append(640)
    obs_bin_counts = np.diff(q_counts)
    log_val = np.log(obs_bin_counts/exp_bin_counts)
    log_val[np.isneginf(log_val)] = 0 

    residual_error = 2*sum(np.array(obs_bin_counts)*log_val) 

    if params[0] < 0 or params[1] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1:
        residual_error = residual_error + 10000000

    return residual_error



#%% Run this to compile fit routines
    
def fit_participant(participant, curvatures, num_fits):
    #train_indices = np.random.choice(640, 576, replace = False)
    for fit_parts in range(num_fits):

        starting_points = np.array([[0.99, 0.1]])
        for initial_point in starting_points:
            if participant%4 == 0 or participant%4 == 1:      
                #fits = scipy.optimize.minimize(residuals_sudden, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3]], args = (640, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1]))), method = 'Nelder-Mead')            
                fits = scipy.optimize.basinhopping(residuals_sudden, x0 = [initial_point[0], initial_point[1]], minimizer_kwargs={'args': (640, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1]))), 'method':'Nelder-Mead'})

                #if fits.fun < fit_V[participant][fit_parts]:
                A = fits.x[0]#fit_Af[participant][fit_parts] = fits.x[0]
                B = fits.x[1]#fit_Bf[participant][fit_parts] = fits.x[1]
                V = fits.fun#fit_V[participant][fit_parts] = fits.fun
                #fit_success[participant][fit_parts] = fits.success                
            else:
                #fits = scipy.optimize.minimize(residuals_gradual, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3]], args = (640, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1]))), method = 'Nelder-Mead')         
                fits = scipy.optimize.basinhopping(residuals_gradual, x0 = [initial_point[0], initial_point[1]], minimizer_kwargs={'args': (640, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1]))), 'method':'Nelder-Mead'})
                #if fits.fun < fit_V[participant][fit_parts]:
                A = fits.x[0]#fit_Af[participant][fit_parts] = fits.x[0]
                B = fits.x[1]#fit_Bf[participant][fit_parts] = fits.x[1]
                V = fits.fun#fit_V[participant][fit_parts] = fits.fun
                #fit_success[participant][fit_parts] = fits.success
            print (participant, V)
    return A, B, V

def run_fits_single(curvatures, num_trials, part_size):
    func = partial(fit_participant, curvatures = curvatures, num_fits = 1)
    pool = Pool()
    res = np.reshape(np.array(pool.map(func, range(60))), (60, 3))
    #return fit_Af, fit_Bf, fit_As, fit_Bs, fit_V
    return res   

#%% Load fit values
"""
fits = pickle.load(open('fit_single_bound_test.pickle', 'rb'))
#fits_1 = pickle.load(open('fit_dual_bound_ontotalrt.pickle', 'rb'))
curvatures_smooth = pickle.load(open('curvatures_smooth.pickle', 'rb'))
curvatures_smooth = curvatures_smooth/90

#%%
participant = 48
if participant%4 == 0 or participant%4 == 1:
    errors = model_sudden(704, fits[participant][0], fits[participant][1])[0]
else:
    errors = model_gradual(704, fits[participant][0], fits[participant][1])[0]
plt.plot(np.ravel(curvatures_smooth[participant][1:]))
plt.plot(errors)


#%% Testing by holding out some data points in the error calculation


#%%
r2_scores_save = np.zeros(60)
#r2_scores_save1 = np.zeros(60)
train_data = np.zeros((60, 576))
train_pred = np.zeros((60, 576))
test_data = np.zeros((60, 64))
test_predictions = np.zeros((60, 64))
test_data1 = np.zeros((60, 64))
test_predictions1 = np.zeros((60, 64))
for participant in range(60):
    mask = np.ones(640, bool)
    mask[fits[participant][3]] = False
    mask_train = np.zeros(640, bool)
    mask_train[fits[participant][3]] = True
    if participant%4 == 0 or participant%4 == 1:
        y_pred = model_sudden(640, fits[participant][0], fits[participant][1])[0]
        y_pred1 = model_sudden(64, fits[participant][0], fits[participant][1])[0]
    else:
        y_pred = model_gradual(640, fits[participant][0], fits[participant][1])[0]
        y_pred1 = model_sudden(64, fits[participant][0], fits[participant][1])[0]
    #r2_scores_save[participant] = r2_score(np.ravel(curvatures_smooth[participant][1:-1]), y_pred)
    train_data[participant] = np.ravel(curvatures_smooth[participant][1:-1])[mask_train]    
    train_pred[participant] = y_pred[mask_train]
    test_data[participant] = np.ravel(curvatures_smooth[participant][1:-1])[mask]
    test_predictions[participant] = y_pred[mask]
    test_data1[participant] = np.ravel(curvatures_smooth[participant][-1])
    test_predictions1[participant] = y_pred1
    #r2_scores_save[participant] = r2_score(test_data1[participant], test_predictions1[participant])
#%%
def plots(test_data, test_predictions, test_data1, test_predictions1):
    fig, axes = plt.subplots(2, 2, sharex = True, sharey=True)
    axes[0, 0].scatter(np.mean(test_data[0::4], axis = 0), np.mean(test_predictions[0::4], axis = 0))
    axes[0, 1].scatter(np.mean(test_data[1::4], axis = 0), np.mean(test_predictions[1::4], axis = 0))
    axes[1, 0].scatter(np.mean(test_data[2::4], axis = 0), np.mean(test_predictions[2::4], axis = 0))
    axes[1, 1].scatter(np.mean(test_data[3::4], axis = 0), np.mean(test_predictions[3::4], axis = 0))
    axes[0, 0].set_title('SS')
    axes[0, 1].set_title('SA')
    axes[1, 0].set_title('GS')
    axes[1, 1].set_title('GA')

    fig.suptitle('Performance on held out Data: Single State Model')
    fig.text(0.5, 0.04, 'Actual Errors', ha='center')
    fig.text(0.04, 0.5, 'Predicted Errors', va='center', rotation='vertical')
    #fig.set_xlabel('Actual Errors')

    fig, axes = plt.subplots(2, 2, sharex=True)
    axes[0, 0].plot(range(64), np.mean(test_data1[0::4], axis = 0), label = 'Actual Data')
    axes[0, 0].plot(range(64), np.mean(test_predictions1[0::4], axis = 0), label = 'Predicted Data')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    axes[0, 1].plot(range(64), np.mean(test_data1[1::4], axis = 0), label = 'Actual Data')
    axes[0, 1].plot(range(64), np.mean(test_predictions1[1::4], axis = 0), label = 'Predicted Data')
    
    axes[1, 0].plot(range(64), np.mean(test_data1[2::4], axis = 0), label = 'Actual Data')
    axes[1, 0].plot(range(64), np.mean(test_predictions1[2::4], axis = 0), label = 'Predicted Data')
    
    axes[1, 1].plot(range(64), np.mean(test_data1[3::4], axis = 0), label = 'Actual Data')
    axes[1, 1].plot(range(64), np.mean(test_predictions1[3::4], axis = 0), label = 'Predicted Data')

    fig.legend(handles, labels, loc='top right', bbox_to_anchor=(0.5, 0.5))
    axes[0, 0].set_title('SS')
    axes[0, 1].set_title('SA')
    axes[1, 0].set_title('GS')
    axes[1, 1].set_title('GA')

    fig.suptitle('Performance on Transfer phase: Single State Model')
    fig.text(0.5, 0.04, 'Transfer Trial Number', ha='center')
    fig.text(0.04, 0.5, 'Transfer Errors', va='center', rotation='vertical')

#plt.scatter(r2_scores_save, r2_scores_save1)


#%%
"""
def main():
    
    #%%Parallelize curvature calculations
        
#    paramlist = list(itertools.product(range(1000, 1060), range(12), range(64), range(1, 2)))
    #if __name__ == '__main__':
    #its = pickle.load(open('its.pickle', 'rb'))
    #mts = pickle.load(open('mts.pickle', 'rb'))
    #total_time = its+mts
    #Test git and vscode 
    #curvatures_smooth = pickle.load(open('dual_with_transfer_generated_errors.pickle', 'rb'))
    #curvatures_smooth = pickle.load(open('generated_by_dual.pickle', 'rb'))
    curvatures_smooth = pickle.load(open('curvatures_smooth.pickle', 'rb'))
    curvatures_smooth = curvatures_smooth/90
    #curvatures_smooth = gaussian_filter1d(total_time, 2)
    #curvatures_smooth = curvatures_smooth/np.max(curvatures_smooth)
    #print("parallel curvatures successful")
    print (curvatures_smooth)
    
    print ("Curvatures Loaded. In Fit routine")

    #%% Parallel run and dump fits
    fits = run_fits_single(curvatures_smooth, 640, 640)
    with open('fit_single_quantiles.pickle', 'wb') as f:
        pickle.dump(fits, f)
    f.close()
        
    
    #%% Run this to save parameters
    
if __name__ == '__main__':
    main()
