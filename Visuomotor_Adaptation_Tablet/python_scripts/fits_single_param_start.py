#!/usr/bin/env python
# coding: utf-8

# In[33]:


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


# In[13]:


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


# In[15]:


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


# In[47]:


def fit_routine(participant, curvature):
    if participant%4 == 0 or participant%4 == 1:        
        fit = scipy.optimize.basinhopping(single_residuals_sudden, x0 = [0.5, 0.5, 0.05], minimizer_kwargs={'args':(640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), 'method' : 'Nelder-Mead'})
    else:
        fit = scipy.optimize.basinhopping(single_residuals_gradual, x0 = [0.5, 0.5, 0.05], minimizer_kwargs={'args':(640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), 'method' : 'Nelder-Mead'})
    A = fit.x[0]
    B = fit.x[1]
    single_sigma = fit.x[2]
    single_neg2ll = fit.fun
    print("Single neg2ll: ", single_neg2ll)
    
    if participant%4 == 0 or participant%4 == 1:        
        fit = scipy.optimize.basinhopping(dual_residuals_sudden, x0 = [A, B, 0.9999, 0.0001, 0.05], minimizer_kwargs={'args' : (640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), 'method' : 'Nelder-Mead'})
    else:
        fit = scipy.optimize.basinhopping(dual_residuals_gradual, x0 = [A, B, 0.9999, 0.0001, 0.05], minimizer_kwargs={'args' : (640, np.nan_to_num(np.ravel(curvature[participant][1:-1]), nan = np.nanmedian(curvature[participant][1:-1]))), 'method' : 'Nelder-Mead'})
    Af = fit.x[0]
    Bf = fit.x[1]
    As = fit.x[2]
    Bs = fit.x[3]

    dual_sigma = fit.x[4]
    dual_neg2ll = fit.fun
    print("Dual neg2ll: ", dual_neg2ll)
    
    return [A, B, single_sigma, single_neg2ll], [Af, Bf, As, Bs, dual_sigma, dual_neg2ll] 

def run_fits_singlestart(curvatures, num_fit_trials):
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
    #print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))
    #print ("Mean Res in dual: ", i, np.mean(res[i][:, -3]))

    return res   

# In[48]:

"""
single_fits = np.zeros((60, 4))
dual_fits = np.zeros((60, 6))
curvature = pickle.load(open('curvatures_smooth.pickle', 'rb'))/90.0
for participant in range(60):
    single_fits[participant], dual_fits[participant] = fit_routine(participant, curvature)
    


# In[29]:


AIC_single = single_fits[:, -1] + 4
AIC_dual = dual_fits[:, -1] + 8


# In[34]:


df = pd.DataFrame({'Participant_ID': np.arange(60), 
                   'Rotation': np.tile(['Sudden', 'Sudden', 'Gradual', 'Gradual'], 15),
                   'Emphasis': np.tile(['Speed', 'Accuracy'], 30),
                   'A': single_fits[:, 0],
                   'B': single_fits[:, 1],
                   'single_sigma': single_fits[:, 2],
                   'single_neg2ll': single_fits[:, 3],
                   
                   'Af': dual_fits[:, 0],
                   'Bf': dual_fits[:, 1],
                   'As': dual_fits[:, 2],
                   'Bs': dual_fits[:, 3],                   
                   'dual_sigma': dual_fits[:, 4],
                   'dual_neg2ll': dual_fits[:, 5],
                   
                   'AIC_single': AIC_single,
                   'AIC_dual': AIC_dual
                  
                  })


# In[37]:


df_selected = pd.DataFrame({'Participant_ID': np.arange(60), 
                            'Rotation': np.tile(['Sudden', 'Sudden', 'Gradual', 'Gradual'], 15),
                            'Emphasis': np.tile(['Speed', 'Accuracy'], 30),
                            'dual_selected_AIC': AIC_dual < AIC_single,
                            'dual_selected_negll': df['dual_neg2ll'] < df['single_neg2ll']
                           })


# In[46]:


df.to_csv('simplex_fits_startparam.csv')


# In[ ]:

"""


