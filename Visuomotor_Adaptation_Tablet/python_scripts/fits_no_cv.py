#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as stat
import scipy.optimize
import math
import pickle
import pandas as pd
import seaborn as sns
from multiprocessing import Pool


# In[2]:


def dual_model_sudden(num_trials, Af, Bf, As, Bs):
    errors = np.zeros((num_trials))
    rotation = 90/90.0
    fast_est = np.zeros((num_trials))
    slow_est = np.zeros((num_trials))
    rotation_est = np.zeros((num_trials))
    #rotation_est[0] = est
    for trial in range(num_trials - 1):
        errors[trial] = rotation - rotation_est[trial]
        #print(errors[trial])
        fast_est[trial+1] = Af*fast_est[trial] + Bf*errors[trial]
        slow_est[trial+1] = As*slow_est[trial] + Bs*errors[trial]
        rotation_est[trial+1] = fast_est[trial+1] + slow_est[trial+1]
        #print (rotation_est)
    errors[num_trials-1] = rotation - rotation_est[num_trials-1]
    return errors, rotation_est, fast_est, slow_est

def dual_model_gradual(num_trials, Af, Bf, As, Bs):
    errors = np.zeros((num_trials))
    fast_est = np.zeros((num_trials))
    slow_est = np.zeros((num_trials))
    rotation_est = np.zeros((num_trials))
    rotation = 0
    for trial in range(num_trials - 1):
        if trial%64 == 0:
            rotation = rotation + 10/90.0
        if rotation > 90/90:
            rotation = 90/90
        errors[trial] = rotation - rotation_est[trial]
        #print(errors[trial])
        fast_est[trial+1] = Af*fast_est[trial] + Bf*errors[trial]
        slow_est[trial+1] = As*slow_est[trial] + Bs*errors[trial]
        rotation_est[trial+1] = fast_est[trial+1] + slow_est[trial+1]
        #print (rotation_est)
    errors[num_trials-1] = rotation - rotation_est[num_trials-1]
    return errors, rotation_est, fast_est, slow_est

def single_model_sudden(num_trials, A, B):
    errors = np.zeros((num_trials))
    rotation = 90/90.0
    rotation_est = np.zeros((num_trials))
    #rotation_est[0] = est
    for trial in range(num_trials - 1):
        errors[trial] = rotation - rotation_est[trial]
        #print(errors[trial])
        rotation_est[trial+1] = A * rotation_est[trial] + B*errors[trial]
        #print (rotation_est)
    errors[num_trials-1] = rotation - rotation_est[num_trials-1]
    return errors, rotation_est

def single_model_gradual(num_trials, A, B):
    errors = np.zeros((num_trials))
    rotation_est = np.zeros((num_trials))
    rotation = 0
    for trial in range(num_trials - 1):
        if trial%64 == 0:
            rotation = rotation + 10/90.0
        if rotation > 90/90:
            rotation = 90/90
        errors[trial] = rotation - rotation_est[trial]
        #print(errors[trial])
        rotation_est[trial+1] = A*rotation_est[trial] + B*errors[trial]
        #print (rotation_est)
    errors[num_trials-1] = rotation - rotation_est[num_trials-1]
    return errors, rotation_est


# In[3]:


def single_residuals_sudden(params, num_trials, data_errors):
    model_errors = single_model_sudden(num_trials, params[0], params[1])[0]
    residual_error = -2*sum(stat.norm.logpdf(data_errors, model_errors, params[2]))
    #residual_error = np.sum(np.square(model_errors - data_errors))

    if params[0] < 0 or params[1] < 0 or params[0] > 1 or params[1] > 1:
        residual_error = residual_error + 10000000
    return residual_error

def single_residuals_gradual(params, num_trials, data_errors):
    model_errors = single_model_gradual(num_trials, params[0], params[1])[0]
    #residual_error = np.sum(np.square(model_errors - data_errors))
    residual_error = -2*sum(stat.norm.logpdf(data_errors, model_errors, params[2]))

    if params[0] < 0 or params[1] < 0 or params[0] > 1 or params[1] > 1:
        residual_error = residual_error + 10000000
    
    return residual_error

def dual_residuals_sudden(params, num_trials, data_errors):
    model_errors = dual_model_sudden(num_trials, params[0], params[1], params[2], params[3])[0]
    #residual_error = np.sum(np.square(model_errors - data_errors))
    residual_error = -2*sum(stat.norm.logpdf(data_errors, model_errors, params[4]))

    if params[0] > params[2]:
        residual_error = residual_error + 10000000
    if params[1] < params[3]:
        residual_error = residual_error + 10000000
    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1 or params[2] > 1 or params[3] > 1:
        residual_error = residual_error + 10000000

    return residual_error

def dual_residuals_gradual(params, num_trials, data_errors):
    model_errors = dual_model_gradual(num_trials, params[0], params[1], params[2], params[3])[0]
    #residual_error = np.sum(np.square(model_errors - data_errors))
    residual_error = -2*sum(stat.norm.logpdf(data_errors, model_errors, params[4]))

    if params[0] > params[2]:
        residual_error = residual_error + 10000000
    if params[1] < params[3]:
        residual_error = residual_error + 10000000
    if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
        residual_error = residual_error + 10000000
    if params[0] > 1 or params[1] > 1 or params[2] > 1 or params[3] > 1:
        residual_error = residual_error + 10000000

    return residual_error


# In[4]:

"""
def fit_routine(participant, curvature):
    single_neg2ll = 1000000
    dual_neg2ll = 1000000
    A = [0.001, 0.01, 0.1, 0.5, 0.9]
    B = [0.001, 0.01, 0.1, 0.5, 0.9]
    sigma = [1, 0.5, 0.05]
    single_starting_points = np.array(np.meshgrid(A, B, sigma)).reshape(3, 75).T
    for i in range(75):
        if participant%4 == 0 or participant%4 == 1:        
            fit = scipy.optimize.minimize(single_residuals_sudden, x0 = [single_starting_points[i][0], single_starting_points[i][1], single_starting_points[i][2]], args = (640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), method = 'Nelder-Mead')
            #fit = scipy.optimize.basinhopping(single_residuals_sudden, x0 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)], minimizer_kwargs={'args':(640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), 'method' : 'Nelder-Mead'})
        else:
            fit = scipy.optimize.minimize(single_residuals_gradual, x0 = x0 = [single_starting_points[i][0], single_starting_points[i][1], single_starting_points[i][2]], args = (640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), method = 'Nelder-Mead')            
            #fit = scipy.optimize.basinhopping(single_residuals_gradual, x0 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)], minimizer_kwargs={'args':(640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), 'method' : 'Nelder-Mead'})
        if fit.fun < single_neg2ll:            
            A = fit.x[0]
            B = fit.x[1]
            single_sigma = fit.x[2]
            single_neg2ll = fit.fun
            print("Participant, i, Single neg2ll: ", participant, i, single_neg2ll)
    Af = [0.001, 0.01, 0.1, 0.5, 0.9]
    Bf = [0.001, 0.01, 0.1, 0.5, 0.9]
    sigma = [1, 0.5, 0.05]
    single_starting_points = np.array(np.meshgrid(A, B, sigma)).reshape(5, 1875).T
    
    for i in range(1875):

        if participant%4 == 0 or participant%4 == 1:        
            fit = scipy.optimize.minimize(dual_residuals_sudden, x0 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)], args = (640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), method = 'Nelder-Mead')
            #fit = scipy.optimize.basinhopping(dual_residuals_sudden, x0 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)], minimizer_kwargs={'args' : (640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), 'method' : 'Nelder-Mead'})
        else:
            fit = scipy.optimize.minimize(dual_residuals_gradual, x0 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)], args = (640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), method = 'Nelder-Mead')            
            #fit = scipy.optimize.basinhopping(dual_residuals_gradual, x0 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)], minimizer_kwargs={'args' : (640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), 'method' : 'Nelder-Mead'})
        if fit.fun < dual_neg2ll:
            Af = fit.x[0]
            Bf = fit.x[1]
            As = fit.x[2]
            Bs = fit.x[3]

            dual_sigma = fit.x[4]
            dual_neg2ll = fit.fun
            print("Participant, i, Dual neg2ll: ", participant, i, dual_neg2ll)
    
    return [A, B, single_sigma, single_neg2ll], [Af, Bf, As, Bs, dual_sigma, dual_neg2ll] 
"""
def single_gridsearch(participant, curvatures):
    num_fit_trials = 640
    #train_length = num_fit_trials - int(np.floor(num_fit_trials/10.0))
    #train_indices = np.random.choice(num_fit_trials, train_length, replace = False)
    #starting_points = np.array([[9.95175980e-01,  4.75713350e-03,  4.69351985e-02], 
    #                           [9.96171356e-01,  1.04296035e-02,  6.94023005e-02], 
    #                           [9.95710143e-01,  9.68652833e-03,  5.04291985e-02], 
    #                           [9.93113960e-01,  3.22839645e-02,  2.69776936e-02]])
    #initial_point = starting_points[participant%4]
    A = np.arange(0.001, 0.99, 0.005)
    B = np.arange(0.001, 0.99, 0.005)
    sigma = np.arange(0.001, 0.99, 0.005)
    starting_points = np.array(np.meshgrid(A, B, sigma)).reshape(3, len(A)*len(B)*len(sigma)).T
    V = 100000
    for initial_point in starting_points:
        if participant%4 == 0 or participant%4 == 1:      
            newV = single_residuals_sudden(initial_point, num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1])))
            if newV < V:
                A = A
                B = B
                epsilon = sigma
                V = newV
        else:
            newV = single_residuals_gradual(initial_point, num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:])))
            if newV < V:
                A = A
                B = B
                epsilon = sigma
                V = newV
    print (participant, V)
    return A, B, V, epsilon

def dual_gridsearch(participant, curvatures):
    num_fit_trials = 640
    #train_length = num_fit_trials - int(np.floor(num_fit_trials/10.0))
    #train_indices = np.random.choice(num_fit_trials, train_length, replace = False)
    #starting_points = np.array([[9.95175980e-01,  4.75713350e-03,  4.69351985e-02], 
    #                           [9.96171356e-01,  1.04296035e-02,  6.94023005e-02], 
    #                           [9.95710143e-01,  9.68652833e-03,  5.04291985e-02], 
    #                           [9.93113960e-01,  3.22839645e-02,  2.69776936e-02]])
    #initial_point = starting_points[participant%4]
    As = np.arange(0.001, 0.99, 0.005)
    Bs = np.arange(0.001, 0.99, 0.005)
    Af = np.arange(0.001, 0.99, 0.005)
    Bf = np.arange(0.001, 0.99, 0.005)                                                 
    sigma = np.arange(0.001, 0.99, 0.005)
    starting_points = np.array(np.meshgrid(A, B, sigma)).reshape(3, len(Af)*len(Bf)*len(As)*len(Bs)*len(sigma)).T
    V = 100000
    for initial_point in starting_points:
        if participant%4 == 0 or participant%4 == 1:      
            newV = dual_residuals_sudden(initial_point, num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1])))
            if newV < V:
                As = As
                Bs = Bs
                Af = Af
                Bf = Bf
                epsilon = sigma
                V = newV
        else:
            newV = dual_residuals_gradual(initial_point, num_fit_trials, np.nan_to_num(np.ravel(curvatures[participant][1:-1]), nan = np.nanmedian(curvatures[participant][1:-1])))
            if newV < V:
                As = As
                Bs = Bs
                Af = Af
                Bf = Bf
                epsilon = sigma
                V = newV
    print (participant, V)
    return Af, Bf, As, Bs, V, epsilon

                                                                 
def run_fits_nocv(curvatures, num_fit_trials):
    #train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    #train_indices = np.array([np.arange(num_participants)])
    pool = Pool()
    #res = np.zeros(num_fits, dtype = object)
    #for i in range(num_fits):
    num_participants = 60
    c_obj = np.zeros(num_participants, dtype = object)
    for participant in range(num_participants):
        c_obj[participant] = curvatures
    participant_args = [x for x in zip(range(num_participants), c_obj[range(num_participants)])]
    res = np.reshape(np.array(pool.starmap(fit_routine, participant_args)), (num_participants, 2))
    #res = np.reshape(np.array(pool.starmap(fit_routine, participant_args)), (num_participants, 2))
    #print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))
    #print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   
                                                                 
def single_gridsearch_run(curvatures, num_fit_trials):
    #train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    #train_indices = np.array([np.arange(num_participants)])
    pool = Pool()
    #res = np.zeros(num_fits, dtype = object)
    #for i in range(num_fits):
    num_participants = 60
    c_obj = np.zeros(num_participants, dtype = object)
    for participant in range(num_participants):
        c_obj[participant] = curvatures
    participant_args = [x for x in zip(range(num_participants), c_obj[range(num_participants)])]
    res = np.reshape(np.array(pool.starmap(single_gridsearch, participant_args)), (num_participants, 4))
    #res = np.reshape(np.array(pool.starmap(fit_routine, participant_args)), (num_participants, 2))
    #res = np.reshape(np.array(pool.starmap(fit_routine, participant_args)), (num_participants, 2))
    #print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))
    #print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   
                                                                 
def dual_gridsearch_run(curvatures, num_fit_trials):
    #train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    #train_indices = np.array([np.arange(num_participants)])
    pool = Pool()
    #res = np.zeros(num_fits, dtype = object)
    #for i in range(num_fits):
    num_participants = 60
    c_obj = np.zeros(num_participants, dtype = object)
    for participant in range(num_participants):
        c_obj[participant] = curvatures
    participant_args = [x for x in zip(range(num_participants), c_obj[range(num_participants)])]
    res = np.reshape(np.array(pool.starmap(dual_gridsearch, participant_args)), (num_participants, 6))
    #res = np.reshape(np.array(pool.starmap(fit_routine, participant_args)), (num_participants, 2))
    #res = np.reshape(np.array(pool.starmap(fit_routine, participant_args)), (num_participants, 2))
    #print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))
    #print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   

                                                                 
