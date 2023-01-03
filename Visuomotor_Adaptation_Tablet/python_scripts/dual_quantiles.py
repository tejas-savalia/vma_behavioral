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

def residuals_sudden(params, num_trials, data_errors):
    model_errors = dual_model_sudden(num_trials, params[0], params[1], params[2], params[3])[0]
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
    if params[0] > params[2]:
        residual_error = residual_error + 10000000
    if params[1] < params[3]:
        residual_error = residual_error + 10000000
    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
        residual_error = residual_error + 10000000
    return residual_error

def residuals_gradual(params, num_trials, data_errors):
    model_errors = dual_model_gradual(num_trials, params[0], params[1], params[2], params[3])[0]
    #residual_error = np.sum((model_errors - data_errors)**2)
    #residual_error = mean_squared_error(data_errors, model_errors)
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

    if params[0] > params[2]:
        residual_error = residual_error + 10000000
    if params[1] < params[3]:
        residual_error = residual_error + 10000000
    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1 or params[2] > 1 or params[3] > 1:
        residual_error = residual_error + 10000000

    return residual_error


#%% Run this to compile fit routines
    
def fit_participant(participant, curvatures, num_fits):

    for fit_parts in range(num_fits):

        starting_points = np.array([[0.9, 0.2, 0.999, 0.01]])
        for initial_point in starting_points:
            if participant%4 == 0 or participant%4 == 1:      
                #fits = scipy.optimize.minimize(residuals_sudden, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3]], args = (640, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1]))), method = 'Nelder-Mead')            
                fits = scipy.optimize.basinhopping(residuals_sudden, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3]], minimizer_kwargs={'args': (640, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1]))), 'method':'Nelder-Mead'})

                #if fits.fun < fit_V[participant][fit_parts]:
                Af = fits.x[0]#fit_Af[participant][fit_parts] = fits.x[0]
                Bf = fits.x[1]#fit_Bf[participant][fit_parts] = fits.x[1]
                As = fits.x[2]#fit_As[participant][fit_parts] = fits.x[2]
                Bs = fits.x[3]#fit_Bs[participant][fit_parts] = fits.x[3]
                V = fits.fun#fit_V[participant][fit_parts] = fits.fun
                #fit_success[participant][fit_parts] = fits.success                
            else:
                #fits = scipy.optimize.minimize(residuals_gradual, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3]], args = (640, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1]))), method = 'Nelder-Mead')         
                fits = scipy.optimize.basinhopping(residuals_gradual, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3]], minimizer_kwargs={'args': (640, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1]))), 'method':'Nelder-Mead'})
                #if fits.fun < fit_V[participant][fit_parts]:
                Af = fits.x[0]#fit_Af[participant][fit_parts] = fits.x[0]
                Bf = fits.x[1]#fit_Bf[participant][fit_parts] = fits.x[1]
                As = fits.x[2]#fit_As[participant][fit_parts] = fits.x[2]
                Bs = fits.x[3]#fit_Bs[participant][fit_parts] = fits.x[3]
                V = fits.fun#fit_V[participant][fit_parts] = fits.fun
                #fit_success[participant][fit_parts] = fits.success
            print (participant, V)
    return Af, Bf, As, Bs, V

def run_fits_dual(curvatures, num_trials, part_size):
    func = partial(fit_participant, curvatures = curvatures, num_fits = 1)
    pool = Pool()
    res = np.reshape(np.array(pool.map(func, range(60))), (60, 5))
    #return fit_Af, fit_Bf, fit_As, fit_Bs, fit_V
    return res   

#%% Load fit values
"""
fits = pickle.load(open('fit_dual_bound_with_transfer.pickle', 'rb'))
#fits_1 = pickle.load(open('fit_dual_bound_ontotalrt.pickle', 'rb'))
curvatures_smooth = pickle.load(open('curvatures_smooth.pickle', 'rb'))
curvatures_smooth = curvatures_smooth/90

#%%
participant = 51
if participant%4 == 0 or participant%4 == 1:
    errors = dual_model_sudden(704, fits[participant][0], fits[participant][1], fits[participant][2], fits[participant][3])[0]
else:
    errors = dual_model_gradual(704, fits[participant][0], fits[participant][1], fits[participant][2], fits[participant][3])[0]
plt.plot(np.ravel(curvatures_smooth[participant][1:]))
plt.plot(errors)
"""
#%%
"""
r2_scores_save = np.zeros(60)
r2_scores_save1 = np.zeros(60)

for participant in range(60):
    if participant%4 == 0 or participant%4 == 1:
        y_pred = dual_model_sudden(640, fits[participant][0], fits[participant][1], fits[participant][2], fits[participant][3])[0]
        y_pred1 = dual_model_sudden(640, fits_1[participant][0], fits_1[participant][1], fits_1[participant][2], fits_1[participant][3])[0]
    else:
        y_pred = dual_model_gradual(640, fits[participant][0], fits[participant][1], fits[participant][2], fits[participant][3])[0]
        y_pred1 = dual_model_gradual(640, fits_1[participant][0], fits_1[participant][1], fits_1[participant][2], fits_1[participant][3])[0]
    r2_scores_save[participant] = r2_score(np.ravel(curvatures_smooth[participant][1:-1]), y_pred)
    r2_scores_save1[participant] = r2_score(np.ravel(curvatures_smooth[participant][1:-1]), y_pred1)

plt.scatter(r2_scores_save, r2_scores_save1)
"""

#%%

def main():
    
    #%%Parallelize curvature calculations
        
#    paramlist = list(itertools.product(range(1000, 1060), range(12), range(64), range(1, 2)))
    #if __name__ == '__main__':
    #its = pickle.load(open('its.pickle', 'rb'))
    #mts = pickle.load(open('mts.pickle', 'rb'))
    curvatures_smooth = pickle.load(open('curvatures_smooth.pickle', 'rb'))
    #curvatures_smooth = pickle.load(open('generated_by_dual.pickle', 'rb'))
    #curvatures_smooth = pickle.load(open('single_with_transfer_generated_errors.pickle', 'rb'))
    #total_time = its+mts
    #Test git and vscode 
    curvatures_smooth = curvatures_smooth/90
    #curvatures_smooth = gaussian_filter1d(total_time, 2)
    #curvatures_smooth = curvatures_smooth/np.max(curvatures_smooth)
    #print("parallel curvatures successful")
    print (curvatures_smooth)
    
    #with open('curvatures_smooth.pickle', 'wb') as f:
    #    pickle.dump(curvatures_smooth, f)
    #f.close()
    print ("Curvatures Loaded. In Fit routine")
    
    #%% Parallel run and dump fits
    fits = run_fits_dual(curvatures_smooth, 640, 640)
    #with open('fit_dual_bound_with_transfer_model_recovery.pickle', 'wb') as f:
    with open('fit_dual_quantiles.pickle', 'wb') as f:
        
        pickle.dump(fits, f)
    f.close()
        
    
    #%% Run this to save parameters
    
if __name__ == '__main__':
    main()
