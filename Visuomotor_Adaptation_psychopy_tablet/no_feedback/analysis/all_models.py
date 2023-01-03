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





def dual_test_fit(participant, curvatures, num_fit_trials, train_indices):
    train_length = num_fit_trials - int(np.floor(num_fit_trials/10.0))
    starting_params = pickle.load(open('fit_dual_CV_640_bestfit_starting_point.pickle', 'rb'))

    #train_indices = np.random.choice(num_fit_trials, train_length, replace = False)
    #Starting points from fits on group average data
    #starting_points = np.array([[5.21669122e-01,  1.73886947e-01,  5.21669136e-01,  1.73886084e-01,
    #    9.99617187e-02], [5.18007232e-01,  2.25765690e-01,  9.98320410e-01,  5.56363200e-03,
    #    3.92042720e-02], [2.65793879e-07,  3.16209104e-01,  9.97065176e-01,  6.42666668e-03,
    #    5.90776470e-02], [2.67461654e-01,  3.05046179e-02,  9.93525435e-01,  3.05045296e-02,
    #    2.72144879e-02]])
    Af = [0.001, 0.1, 0.9]
    Bf = [0.001, 0.1, 0.9]
    As = [0.001, 0.1, 0.9]
    Bs = [0.001, 0.1, 0.9]
    sigma = [1, 0.05]
    #starting_points = np.array(np.meshgrid(Af, Bf, As, Bs, sigma)).reshape(5, 3*3*3*3*2).T
    #initial_point = starting_points[participant%4]
    starting_point = starting_params[participant]
    #print(starting_point)
    #V = 100000
    #for initial_point in starting_points:
    if participant%2 == 0:      
        fits = scipy.optimize.minimize(dual_residuals_sudden, x0 = [starting_point[0], starting_point[1], starting_point[2], starting_point[3], starting_point[5]], args = (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), method = 'Nelder-Mead')
        #if fits.fun < V:
        Af = fits.x[0]
        Bf = fits.x[1]
        As = fits.x[2]
        Bs = fits.x[3]
        epsilon = fits.x[4]
        V = fits.fun
    else:
        fits = scipy.optimize.minimize(dual_residuals_gradual, x0 = [starting_point[0], starting_point[1], starting_point[2], starting_point[3], starting_point[5]], args = (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), method = 'Nelder-Mead')
        #if fits.fun < V:
        Af = fits.x[0]
        Bf = fits.x[1]
        As = fits.x[2]
        Bs = fits.x[3]
        epsilon = fits.x[4]
        V = fits.fun

    print (participant, V)
    return Af, Bf, As, Bs, V, epsilon, train_indices


def single_test_fit(participant, curvatures, num_fit_trials, train_indices):
    train_length = num_fit_trials - int(np.floor(num_fit_trials/10.0))
    starting_params = pickle.load(open('fit_single_CV_640_bestfit_starting_point.pickle', 'rb'))

    #train_indices = np.random.choice(num_fit_trials, train_length, replace = False)
    #starting_points = np.array([[9.95175980e-01,  4.75713350e-03,  4.69351985e-02], 
    #                           [9.96171356e-01,  1.04296035e-02,  6.94023005e-02], 
    #                           [9.95710143e-01,  9.68652833e-03,  5.04291985e-02], 
    #                           [9.93113960e-01,  3.22839645e-02,  2.69776936e-02]])
    #initial_point = starting_points[participant%4]
    #A = [0.001, 0.1, 0.9]
    B = [0.001, 0.1, 0.9]
    sigma = [1, 0.05]
    starting_point = starting_params[participant]
    V = 100000
    #for initial_point in starting_points:
    if participant%4 == 0 or participant%4 == 1:      
        fits = scipy.optimize.minimize(single_residuals_sudden, x0 = [starting_point[0], starting_point[1], starting_point[3]], args =  (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), method = 'Nelder-Mead')
        if fits.fun < V:
            A = fits.x[0]
            B = fits.x[1]
            epsilon = fits.x[2]
            V = fits.fun
    else:
        fits = scipy.optimize.minimize(single_residuals_gradual, x0 = [starting_point[0], starting_point[1], starting_point[3]], args = (num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:])), train_indices), method = 'Nelder-Mead')
        if fits.fun < V:
            A = fits.x[0]
            B = fits.x[1]
            epsilon = fits.x[2]
            V = fits.fun
    print (participant, V)
    return A, B, V, epsilon, train_indices





# # Running Fit routines




def run_fits_single(curvatures, num_fit_trials, num_fits, num_participants):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))#.astype(int)
    #train_indices = np.array([np.arange(704)])

    #print(train_indices[0].shape)
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
#Change 400 to 60 for normal fits
        c_obj = np.zeros(num_participants, dtype = object)
        for participant in range(num_participants):
            c_obj[participant] = curvatures
        participant_args = [x for x in zip(range(num_participants), c_obj[range(num_participants)],  np.repeat(num_fit_trials, num_participants), np.tile(train_indices[i], 64))]
        res[i] = np.reshape(np.array(pool.starmap(single_test_fit, participant_args)), (num_participants, 5))
        print ("Mean Res in Single: ", i, np.mean(res[i][:, -3]))
    return res   

def run_fits_dual(curvatures, num_fit_trials, num_fits, num_participants):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))#.astype(int)
    #starting_params = pickle.load(open('dual_start_params_forcv.pickle', 'rb'))
    #print(starting_params.shape)

    #train_indices = np.array([np.arange(num_participants)])
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
        c_obj = np.zeros(num_participants, dtype = object)
        for participant in range(num_participants):
            c_obj[participant] = curvatures
        participant_args = [x for x in zip(range(num_participants), c_obj[range(num_participants)],  np.repeat(num_fit_trials, num_participants), np.tile(train_indices[i], 64))]
        res[i] = np.reshape(np.array(pool.starmap(dual_test_fit, participant_args)), (num_participants, 7))
        print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   


