#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from multiprocessing import Pool
from functools import partial
import pickle
import scipy
import scipy.optimize
from sklearn.metrics import *
import scipy.stats as stat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# # Dual Models

# ## With Transfer (4 Params)

# In[2]:

def hybrid_sudden(num_trials, A, B, Af, Bf, As, Bs):
    errors_dual = np.zeros((num_trials))
    errors_single = np.zeros((num_trials))    
    rotation = 1.0
    fast_est = np.zeros((num_trials))
    slow_est = np.zeros((num_trials))
    dual_est = np.zeros((num_trials))
    single_est = np.zeros((num_trials))
    #rotation_est[0] = est
    for trial in range(num_trials - 1):
        if trial < 640:
            rotation = 1.0
            errors_dual[trial] = rotation - dual_est[trial]
            errors_single[trial] = rotation - single_est[trial]
        
            fast_est[trial+1] = Af*fast_est[trial] + Bf*errors_dual[trial]
            slow_est[trial+1] = As*slow_est[trial] + Bs*errors_dual[trial]
            single_est[trial+1] = A*single_est[trial] + B*errors_single[trial]

        else:
            rotation = 0
            errors_dual[trial] = errors_dual[trial]
            errors_single[trial] = errors_single[trial]

            #print(errors[trial])
            fast_est[trial+1] = Af*fast_est[trial] - Bf*errors_dual[trial]
            slow_est[trial+1] = As*slow_est[trial] - Bs*errors_dual[trial]
            single_est[trial+1] = A*single_est[trial] - B*errors_single[trial]

        dual_est[trial+1] = fast_est[trial+1] + slow_est[trial+1]
        
        #print (rotation_est)
    errors_dual[num_trials-1] = dual_est[num_trials-1]
    errors_single[num_trials-1] = single_est[num_trials-1]
    
    
    return errors_single, errors_dual, single_est, dual_est


def hybrid_gradual(num_trials, A, B, Af, Bf, As, Bs):
    errors_dual = np.zeros((num_trials))
    errors_single = np.zeros((num_trials))
    fast_est = np.zeros((num_trials))
    slow_est = np.zeros((num_trials))
    dual_est = np.zeros((num_trials))
    single_est = np.zeros((num_trials))
    rotation = 0
    for trial in range(num_trials - 1):
        if trial < 640:
            if trial%64 == 0:
                rotation = rotation + 10/90.0
            if rotation > 1.0:
                rotation = 1.0
            errors_dual[trial] = rotation - dual_est[trial]
            errors_single[trial] = rotation - single_est[trial]
            fast_est[trial+1] = Af*fast_est[trial] + Bf*errors_dual[trial]
            slow_est[trial+1] = As*slow_est[trial] + Bs*errors_dual[trial]
            single_est[trial+1] = A*single_est[trial] + B*errors_single[trial]

        else:
            rotation = 0
            errors_dual[trial] = dual_est[trial] 
            errors_single[trial] = single_est[trial]
            fast_est[trial+1] = Af*fast_est[trial] - Bf*errors_dual[trial]
            slow_est[trial+1] = As*slow_est[trial] - Bs*errors_dual[trial]
            single_est[trial+1] = A*single_est[trial] - B*errors_single[trial]


        dual_est[trial+1] = fast_est[trial+1] + slow_est[trial+1]
        #print (rotation_est)
        
    errors_dual[num_trials-1] = dual_est[num_trials-1]
    errors_single[num_trials-1] = single_est[num_trials-1]


    return errors_single, errors_dual, single_est, dual_est



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
            #slow_est[trial+1] = As*slow_est[trial] + Bs*(errors[trial] - Af*fast_est[trial])
            
        else:
            rotation = 0
            errors[trial] = rotation_est[trial]
        #print(errors[trial])
            fast_est[trial+1] = Af*fast_est[trial] - Bf*errors[trial]
            slow_est[trial+1] = As*slow_est[trial] - Bs*errors[trial]
            #slow_est[trial+1] = As*slow_est[trial] - Bs*(errors[trial] - Af*fast_est[trial])

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
            #slow_est[trial+1] = As*slow_est[trial] + Bs*(errors[trial] - Af*fast_est[trial])

        else:
            rotation = 0
            errors[trial] = rotation_est[trial] 
            fast_est[trial+1] = Af*fast_est[trial] - Bf*errors[trial]
            slow_est[trial+1] = As*slow_est[trial] - Bs*errors[trial]
            #slow_est[trial+1] = As*slow_est[trial] - Bs*(errors[trial] - Af*fast_est[trial])

        rotation_est[trial+1] = fast_est[trial+1] + slow_est[trial+1]
        #print (rotation_est)
    errors[num_trials-1] = rotation_est[num_trials-1]

    return errors, rotation_est, fast_est, slow_est

def dual_model_sudden_alpha(num_trials, Af, Bf, As, Bs, alpha):
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

        rotation_est[trial+1] = alpha*fast_est[trial+1] + (2 - alpha)*slow_est[trial+1]
        #print (rotation_est)
    errors[num_trials-1] = rotation_est[num_trials-1]
    return errors, rotation_est, fast_est, slow_est


def dual_model_gradual_alpha(num_trials, Af, Bf, As, Bs, alpha):
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

        rotation_est[trial+1] = alpha*fast_est[trial+1] + (2 - alpha)*slow_est[trial+1]
        #print (rotation_est)
    errors[num_trials-1] = rotation_est[num_trials-1]

    return errors, rotation_est, fast_est, slow_est


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


def single_residuals_sudden(params, num_trials, data_errors, train_indices):
    #print(train_indices)
    model_errors = model_sudden(num_trials, params[0], params[1])[0]
    model_errors_train = np.take(model_errors, train_indices[train_indices < len(model_errors)])
    data_errors_train = np.take(data_errors, train_indices[train_indices < len(model_errors)])
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[2]))

    if params[0] < 0 or params[1] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1:
        residual_error = residual_error + 10000000

    return residual_error

def single_residuals_gradual(params, num_trials, data_errors, train_indices):
    
    model_errors = model_gradual(num_trials, params[0], params[1])[0]
    model_errors_train = np.take(model_errors, train_indices[train_indices < len(model_errors)])
    data_errors_train = np.take(data_errors, train_indices[train_indices < len(model_errors)])
    #residual_error = np.sum(np.square(model_errors_ train - data_errors_train))
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[2]))
    if params[0] < 0 or params[1] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1:
        residual_error = residual_error + 10000000

    return residual_error

def dual_residuals_sudden(params, num_trials, data_errors, train_indices):
    model_errors = dual_model_sudden(num_trials, params[0], params[1], params[2], params[3])[0]
    model_errors_train = np.take(model_errors, train_indices[train_indices < len(model_errors)])
    data_errors_train = np.take(data_errors, train_indices[train_indices < len(model_errors)])
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[4]))
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    if params[0] > params[2]:
        residual_error = residual_error + 10000000
    if params[1] < params[3]:
        residual_error = residual_error + 10000000
    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1 or params[2] > 1 or params[3] > 1:
        residual_error = residual_error + 10000000

    return residual_error

def dual_residuals_gradual(params, num_trials, data_errors, train_indices):
    model_errors = dual_model_gradual(num_trials, params[0], params[1], params[2], params[3])[0]
    model_errors_train = np.take(model_errors, train_indices[train_indices < len(model_errors)])
    data_errors_train = np.take(data_errors, train_indices[train_indices < len(model_errors)])
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[4]))
    if params[0] > params[2]:
        residual_error = residual_error + 10000000
    if params[1] < params[3]:
        residual_error = residual_error + 10000000
    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1 or params[2] > 1 or params[3] > 1:
        residual_error = residual_error + 10000000

    return residual_error

def dual_residuals_sudden_alpha(params, num_trials, data_errors, train_indices, fit_dual_params):
    model_errors = dual_model_sudden_alpha(num_trials, fit_dual_params[0], fit_dual_params[1], fit_dual_params[2], fit_dual_params[3], params[0])[0]
    model_errors_train = np.take(model_errors, train_indices[train_indices < len(model_errors)])
    data_errors_train = np.take(data_errors, train_indices[train_indices < len(model_errors)])
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[1]))
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    if params[0] > 1:
        residual_error = residual_error + 10000000
#    if params[1] < params[3]:
#        residual_error = residual_error + 10000000
#    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
#        residual_error = residual_error + 10000000
#    if params[0] > 1 or params[1] > 1 or params[2] > 1 or params[3] > 1:
#        residual_error = residual_error + 10000000
#    if params[4] < 0 or params[4] > 2:
#        residual_error = residual_error + 10000000

    return residual_error

def dual_residuals_gradual_alpha(params, num_trials, data_errors, train_indices, fit_dual_params):
    #model_errors = dual_model_gradual(num_trials, params[0], params[1], params[2], params[3])[0]
    model_errors = dual_model_gradual_alpha(num_trials, fit_dual_params[0], fit_dual_params[1], fit_dual_params[2], fit_dual_params[3], params[0])[0]
    model_errors_train = np.take(model_errors, train_indices[train_indices < len(model_errors)])
    data_errors_train = np.take(data_errors, train_indices[train_indices < len(model_errors)])
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[1]))
    if params[0] > 1:
        residual_error = residual_error + 10000000
#    if params[0] > params[2]:
#        residual_error = residual_error + 10000000
#    if params[1] < params[3]:
#        residual_error = residual_error + 10000000
#    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
#        residual_error = residual_error + 10000000
#    if params[0] > 1 or params[1] > 1 or params[2] > 1 or params[3] > 1:
#        residual_error = residual_error + 10000000
#    if params[4] < 0 or params[4] > 2:
#        residual_error = residual_error + 10000000

    return residual_error

def hybrid_residuals_sudden(params, num_trials, data_errors, train_indices, single_params, dual_params):

    model_errors_single = model_sudden(num_trials, single_params[0], single_params[1])[0]
    model_errors_dual = dual_model_sudden(num_trials, dual_params[0], dual_params[1], dual_params[2], dual_params[3])[0]
    model_errors = params[0]*model_errors_single + (1 - params[0])*model_errors_dual

    model_errors_train = np.take(model_errors, train_indices[train_indices < len(model_errors)])
    data_errors_train = np.take(data_errors, train_indices[train_indices < len(model_errors)])
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[1]))
    if params[0] < 0 or params[0] > 1:
        residual_error = residual_error + 1000000
    return residual_error

def hybrid_residuals_gradual(params, num_trials, data_errors, train_indices, single_params, dual_params):
    model_errors_dual = dual_model_gradual(num_trials, dual_params[0], dual_params[1], dual_params[2], dual_params[3])[0]
    model_errors_single = model_gradual(num_trials, single_params[0], single_params[1])[0]
    model_errors = params[0]*model_errors_single + (1 - params[0])*model_errors_dual

    model_errors_train = np.take(model_errors, train_indices[train_indices < len(model_errors)])
    data_errors_train = np.take(data_errors, train_indices[train_indices < len(model_errors)])
    #residual_error = np.sum(np.square(model_errors_train - data_errors_train))
    residual_error = -2*sum(stat.norm.logpdf(data_errors_train, model_errors_train, params[1]))
    if params[0] < 0 or params[0] > 1:
        residual_error = residual_error + 1000000
    return residual_error



def dual_test_fit(participant, curvatures, num_fit_trials, train_indices):
    train_length = num_fit_trials - int(np.floor(num_fit_trials/10.0))
    
    #train_indices = np.random.choice(num_fit_trials, train_length, replace = False)
    starting_points = np.array([[0.9, 0.3, 0.99, 0.01, 0.05]])
    for initial_point in starting_points:
        if participant%4 == 0 or participant%4 == 1:      
            fits = scipy.optimize.basinhopping(dual_residuals_sudden, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3], initial_point[4]], minimizer_kwargs={'args': (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), 'method':'Nelder-Mead'})

            Af = fits.x[0]
            Bf = fits.x[1]
            As = fits.x[2]
            Bs = fits.x[3]
            epsilon = fits.x[4]
            V = fits.fun
        else:
            fits = scipy.optimize.basinhopping(dual_residuals_gradual, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3], initial_point[4]], minimizer_kwargs={'args': (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), 'method':'Nelder-Mead'})
            Af = fits.x[0]
            Bf = fits.x[1]
            As = fits.x[2]
            Bs = fits.x[3]
            epsilon = fits.x[4]
            V = fits.fun
            
        print (participant, V)
    return Af, Bf, As, Bs, V, epsilon, train_indices

def dual_alpha_test_fit(participant, curvatures, num_fit_trials, train_indices, fit_dual_params):
    print('In dual alpha, participant: ', participant)
    train_length = num_fit_trials - int(np.floor(num_fit_trials/10.0))
    
    #train_indices = np.random.choice(num_fit_trials, train_length, replace = False)
    starting_points = np.array([[0.5, 0.05]])
    for initial_point in starting_points:
        if participant%4 == 0 or participant%4 == 1:      
            fits = scipy.optimize.basinhopping(dual_residuals_sudden_alpha, x0 = [initial_point[0], initial_point[1]], minimizer_kwargs={'args': (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices, fit_dual_params), 'method':'Nelder-Mead'})

            alpha = fits.x[0]

            epsilon = fits.x[1]
            V = fits.fun
        else:
            fits = scipy.optimize.basinhopping(dual_residuals_gradual_alpha, x0 = [initial_point[0], initial_point[1]], minimizer_kwargs={'args': (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices, fit_dual_params), 'method':'Nelder-Mead'})

            alpha = fits.x[0]

            epsilon = fits.x[1]
            V = fits.fun
            
        print (participant, V)
    return alpha, V, epsilon, train_indices


def single_test_fit(participant, curvatures, num_fit_trials, train_indices):
    train_length = num_fit_trials - int(np.floor(num_fit_trials/10.0))
    #train_indices = np.random.choice(num_fit_trials, train_length, replace = False)
    starting_points = np.array([[0.9, 0.2, 0.5]])
    for initial_point in starting_points:
        if participant%4 == 0 or participant%4 == 1:      
            fits = scipy.optimize.basinhopping(single_residuals_sudden, x0 = [initial_point[0], initial_point[1], initial_point[2]], minimizer_kwargs={'args': (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), 'method':'Nelder-Mead'})

            A = fits.x[0]
            B = fits.x[1]
            epsilon = fits.x[2]
            V = fits.fun
        else:
            fits = scipy.optimize.basinhopping(single_residuals_gradual, x0 = [initial_point[0], initial_point[1], initial_point[2]], minimizer_kwargs={'args': (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), 'method':'Nelder-Mead'})
            
            A = fits.x[0]
            B = fits.x[1]
            epsilon = fits.x[2]
            V = fits.fun
        print (participant, V)
    return A, B, V, epsilon, train_indices

def hybrid_test_fit(participant, curvatures, num_fit_trials, train_indices, best_single, best_dual):
    train_length = num_fit_trials - int(np.floor(num_fit_trials/10.0))
    
    #train_indices = np.random.choice(num_fit_trials, train_length, replace = False)
    #starting_points = np.array([[0.9, 0.2, 0.9, 0.3, 0.99, 0.01, 0.5, 0.5]])
    starting_points = np.array([[0.5]])
    for initial_point in starting_points:
        if participant%4 == 0 or participant%4 == 1:      
            fits = scipy.optimize.basinhopping(hybrid_residuals_sudden, x0 = [initial_point[0]], minimizer_kwargs={'args': (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices, best_single[participant][:2], best_dual[participant][:4]), 'method':'Nelder-Mead'})
            #fits = scipy.optimize.minimize(hybrid_residuals_sudden, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3], initial_point[4], initial_point[5], initial_point[6], initial_point[7]], args = (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), method = 'Nelder-Mead')


            alpha = fits.x[0]

            V = fits.fun
        else:
            fits = scipy.optimize.basinhopping(hybrid_residuals_gradual, x0 = [initial_point[0]], minimizer_kwargs={'args': (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices, best_single[participant][:2], best_dual[participant][:4]), 'method':'Nelder-Mead'})

            #fits = scipy.optimize.minimize(hybrid_residuals_gradual, x0 = [initial_point[0], initial_point[1], initial_point[2], initial_point[3], initial_point[4], initial_point[5], initial_point[6], initial_point[7]], args = (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), method = 'Nelder-Mead')

            alpha = fits.x[0]

            V = fits.fun
    print (participant, V)
    return alpha, V, train_indices



# # Running Fit routines

# In[8]:


def run_fits_dual(curvatures, num_fit_trials, num_fits):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    train_indices = np.hstack((train_indices, train_indices, train_indices, train_indices))
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
        c_obj = np.zeros(60, dtype = object)
        for participant in range(60):
            c_obj[participant] = curvatures
        participant_args = [x for x in zip(range(60), c_obj[range(60)],  np.repeat(num_fit_trials, 60), train_indices[i])]
        res[i] = np.reshape(np.array(pool.starmap(dual_test_fit, participant_args)), (60, 7))
        print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   

def run_fits_dual_alpha(curvatures, num_fit_trials, num_fits):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    fit_dual_params = pickle.load(open('fit_dual_CV_704.pickle', 'rb'))
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
        c_obj = np.zeros(60, dtype = object)
        for participant in range(60):
            c_obj[participant] = curvatures
        participant_args = [x for x in zip(range(60), c_obj[range(60)],  np.repeat(num_fit_trials, 60), train_indices[i], fit_dual_params[i])]
        res[i] = np.reshape(np.array(pool.starmap(dual_alpha_test_fit, participant_args)), (60, 4))
        print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   


def run_fits_single(curvatures, num_fit_trials, num_fits, num_participants):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    #train_indices = np.array([np.arange(704)])

    print(train_indices[0].shape)
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
#Change 400 to 60 for normal fits
        c_obj = np.zeros(num_participants, dtype = object)
        for participant in range(num_participants):
            c_obj[participant] = curvatures
        participant_args = [x for x in zip(range(num_participants), c_obj[range(num_participants)],  np.repeat(num_fit_trials, num_participants), train_indices[i])]
        res[i] = np.reshape(np.array(pool.starmap(single_test_fit, participant_args)), (num_participants, 5))
        print ("Mean Res in Single: ", i, np.mean(res[i][:, -3]))
    return res   

def run_fits_dual(curvatures, num_fit_trials, num_fits, num_participants):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    #train_indices = np.array([np.arange(num_participants)])
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
        c_obj = np.zeros(num_participants, dtype = object)
        for participant in range(num_participants):
            c_obj[participant] = curvatures
        participant_args = [x for x in zip(range(num_participants), c_obj[range(num_participants)],  np.repeat(num_fit_trials, num_participants), train_indices[i])]
        res[i] = np.reshape(np.array(pool.starmap(dual_test_fit, participant_args)), (num_participants, 7))
        print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   

def run_fits_hybrid(curvatures, num_fit_trials, num_fits):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    best_single = pickle.load(open('fit_single_CV_704.pickle', 'rb'))
    best_dual = pickle.load(open('fit_dual_CV_704.pickle', 'rb'))
    
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
        c_obj = np.zeros(60, dtype = object)
        for participant in range(60):
            c_obj[participant] = curvatures
            
        participant_args = [x for x in zip(range(60), c_obj[range(60)],  np.repeat(num_fit_trials, 60), train_indices[i], best_single, best_dual)]
        res[i] = np.reshape(np.array(pool.starmap(hybrid_test_fit, participant_args)), (60, 4))
        print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   
