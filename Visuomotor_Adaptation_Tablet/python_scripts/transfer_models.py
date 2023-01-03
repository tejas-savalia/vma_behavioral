#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import scipy.stats as stat
from scipy import stats
import pickle
import pandas as pd
import seaborn as sns
sns.set_theme()
import statsmodels


# In[14]:




# In[31]:


def single_transfer(num_trials, A, B, prev_est):
    error = np.zeros(num_trials)
    rotation_est = np.zeros(num_trials)
    rotation_est[0] = prev_est
    for trial in range(num_trials-1):
        error[trial] = rotation_est[trial]
        rotation_est[trial+1] = A*rotation_est[trial] - B*error[trial]
    error[trial+1] = rotation_est[trial+1]
    return error


# In[32]:


#prev_est is a list [fast_est, slow_est]
def dual_transfer(num_trials, Af, Bf, As, Bs, prev_est):
    error = np.zeros(num_trials)
    rotation_est = np.zeros(num_trials)
    fast_est = prev_est[0]
    slow_est = prev_est[1]
    rotation_est[0] = fast_est+slow_est
    for trial in range(num_trials-1):
        error[trial] = rotation_est[trial]
        fast_est = Af*fast_est - Bf*error[trial]
        slow_est = As*slow_est - Bs*error[trial]
        rotation_est[trial+1] = fast_est + slow_est
    error[trial+1] = rotation_est[trial+1]
    return error


# In[67]:


def single_transfer_residuals(params, num_trials, data_errors, prev_est, train_indices):
    A = params[0]
    B = params[1]
    epsilon = params[2]
    model_errors = single_transfer(num_trials, A, B, prev_est)
    model_errors_train = model_errors[np.sort(train_indices)]
    data_errors_train = data_errors[np.sort(train_indices)]
    #plt.plot(model_errors)
    #plt.plot(data_errors_train)
    residuals = -2*np.sum(stats.norm.logpdf(data_errors_train, model_errors_train, epsilon))
    if A < 0 or B < 0 or A > 1 or B > 1:
        residuals = residuals + 100000000
    return residuals


# In[68]:


def dual_transfer_residuals(params, num_trials, data_errors, prev_est, train_indices):
    Af = params[0]
    Bf = params[1]
    As = params[2]
    Bs = params[3]
    epsilon = params[4]
    model_errors = dual_transfer(num_trials, Af, Bf, As, Bs, prev_est)
    model_errors_train = model_errors[np.sort(train_indices)]
    data_errors_train = data_errors[np.sort(train_indices)]
    #plt.plot(model_errors)
    #plt.plot(data_errors_train)
    residuals = -2*np.sum(stats.norm.logpdf(data_errors_train, model_errors_train, epsilon))
    if Af < 0 or Bf < 0 or Af > 1 or Bf > 1 or As < 0 or Bs < 0 or As > 1 or Bs > 1:
        residuals = residuals + 100000000
    if Bf < Bs or Af > As:
        residuals = residuals + 100000000

    return residuals


# In[87]:


def fit_single_transfer(participant, curvatures, training_indices):
    ti = training_indices[training_indices > 640] - 640
    fits = scipy.optimize.basinhopping(single_transfer_residuals, x0 = [0.8, 0.1, 0.1], 
                                       minimizer_kwargs={'args': (64, 
                                                             curvatures[participant][-1], 
                                                             1 - np.nanmean(curvatures[participant][10][-16:]), 
                                                             ti),
                                                         'method': 'Nelder-Mead'})
    A = fits.x[0]
    B = fits.x[1]
    epsilon = fits.x[2]
    V = fits.fun
    print (participant, V)
    return A, B, V, epsilon, ti

def fit_dual_transfer(participant, curvatures, training_indices, fast_est, slow_est):
    ti = training_indices[training_indices > 640] - 640
    #print (ti.shape)
    #print (fast_est.shape)
    fits = scipy.optimize.basinhopping(dual_transfer_residuals, x0 = [0.8, 0.2, 0.9, 0.1, 0.1], 
                                  minimizer_kwargs={'args': (64, 
                                                             curvatures[participant][-1], 
                                                             [fast_est, slow_est], 
                                                             ti),
                                                   'method': 'Nelder-Mead'} )    
    Af = fits.x[0]
    Bf = fits.x[1]
    As = fits.x[2]
    Bs = fits.x[3]
    epsilon = fits.x[4]
    V = fits.fun
    print (participant, V)
    return Af, Bf, As, Bs, V, epsilon, ti


# In[ ]:


def run_fits_single_transfer(curvatures, num_fit_trials, num_fits):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    print(train_indices[0].shape)
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
        c_obj = np.zeros(60, dtype = object)
        for participant in range(60):
            c_obj[participant] = curvatures
        participant_args = [x for x in zip(range(60), c_obj[range(60)], train_indices[i])]
        res[i] = np.reshape(np.array(pool.starmap(fit_single_transfer, participant_args)), (60, 5))
        print ("Mean Res in Single Transfer: ", i, np.mean(res[i][:, -3]))
    return res   

def run_fits_dual_transfer(curvatures, num_fit_trials, num_fits):
    train_indices = pickle.load(open('train_indices_704.pickle', 'rb'))
    fast_est = pickle.load(open('fast_est.pickle', 'rb'))
    slow_est = pickle.load(open('slow_est.pickle', 'rb'))
    pool = Pool()
    res = np.zeros(num_fits, dtype = object)
    for i in range(num_fits):
        c_obj = np.zeros(60, dtype = object)
        for participant in range(60):
            c_obj[participant] = curvatures
        participant_args = [x for x in zip(range(60), c_obj[range(60)],  train_indices[i], fast_est[i], slow_est[i])]
        res[i] = np.reshape(np.array(pool.starmap(fit_dual_transfer, participant_args)), (60, 7))
        print ("Mean Res in dual transfer: ", i, np.mean(res[i][:, -3]))

    return res   


# In[88]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




