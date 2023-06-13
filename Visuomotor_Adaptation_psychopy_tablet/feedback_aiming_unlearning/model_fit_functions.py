import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stat
from scipy.optimize import minimize
import itertools
import multiprocessing as mp

def single_state_model(A, B, num_trials, p_type):
    rotation_estimate = np.zeros(num_trials)
    error = np.zeros(num_trials)
    if p_type == 'Sudden':
        rotation = np.pi/3
        for trial in range(1, num_trials):
            error[trial-1] = rotation - rotation_estimate[trial-1]
            rotation_estimate[trial] = A*rotation_estimate[trial-1] + B*error[trial-1]
    else:
        rotation = np.pi/18
        for trial in range(1, num_trials):
            error[trial-1] = rotation - rotation_estimate[trial-1]
            rotation_estimate[trial] = A*rotation_estimate[trial-1] + B*error[trial-1]
            if trial%64 == 0:
                if rotation < np.pi/3:
                    rotation = rotation + np.pi/18
    error[trial] = rotation - rotation_estimate[trial]
    return error

def dual_state_model(As, Bs, Af, Bf, num_trials, p_type):
    rotation_estimate = np.zeros(num_trials)
    fast_estimate = np.zeros(num_trials)
    slow_estimate = np.zeros(num_trials)
    
    error = np.zeros(num_trials)
    if p_type == 'Sudden':
        rotation = np.pi/3

        for trial in range(1, num_trials):
            error[trial-1] = rotation - rotation_estimate[trial-1]
            fast_estimate[trial] = Af*fast_estimate[trial-1] + Bf*error[trial-1]
            slow_estimate[trial] = As*slow_estimate[trial-1] + Bs*error[trial-1]
            rotation_estimate[trial] = fast_estimate[trial] + slow_estimate[trial]
            
    else:
        rotation = np.pi/18
        for trial in range(1, num_trials):
            error[trial-1] = rotation - rotation_estimate[trial-1]
            fast_estimate[trial] = Af*fast_estimate[trial-1] + Bf*error[trial-1]
            slow_estimate[trial] = As*slow_estimate[trial-1] + Bs*error[trial-1]
            rotation_estimate[trial] = fast_estimate[trial] + slow_estimate[trial]

            if trial%64 == 0:
                if rotation < np.pi/3:
                    rotation = rotation + np.pi/18
    error[trial] = rotation - rotation_estimate[trial-1]

    return error

def calc_log_likelihood(params, data, model, p_type):
    if model == 'single state':
        if any(params[:-1]) < 0 or any(params) > 1:
            return np.inf
        model_pred = single_state_model(params[0], params[1], len(data), p_type)
    else:
        if any(params[:-1]) < 0 or any(params) > 1 or params[0] < params[2] or params[1] > params[3]:
            return np.inf        
        model_pred = dual_state_model(params[0], params[1], params[2], params[3], len(data), p_type)
    log_lik = sum(stat.norm.logpdf(data, model_pred, params[-1]))
    return -log_lik


############ Fitting Functions
#Load Data
data = pd.read_csv('learning_10pcutoff_errors.csv')


def fit_single(participant):
    try:
        errors = data.loc[data['p_id'] == participant, 'errors'].values#*np.pi/180
        p_type = data.loc[data['p_id'] == participant, 'Rotation'].unique()
        curr_fitval = np.inf
        possible_starting_points = itertools.product(np.linspace(0, 1, 8), np.linspace(0, 1, 8), np.linspace(0, 1, 8))
        for i in possible_starting_points:
            temp_res = minimize(calc_log_likelihood, x0=i, args=(errors, 'single state', p_type), bounds=((0, 1), (0, 1), (0, 1)), method = 'Nelder-Mead')
            if temp_res.fun < curr_fitval:
                res = temp_res
                curr_fitval = res.fun
        print('participant done: ', participant)
    except:
        print('participant failed: ', participant)
        return participant, np.nan, np.nan, np.nan, np.nan
    return participant, res.fun, res.x[0], res.x[1], res.x[2]

#load single fits to use as slow learning starting points.
single_fits = pd.read_csv('single_fit_10pcutofferror_results.csv').drop('Unnamed: 0', axis = 1).reset_index().drop('index', axis = 1)

def fit_dual(participant):
    try:
        errors = data.loc[data['p_id'] == participant, 'errors'].values*np.pi/180
        p_type = data.loc[data['p_id'] == participant, 'Rotation'].unique()
        As_init = single_fits.loc[single_fits['p_id'] == participant, 'A'].values[0]
        Bs_init = single_fits.loc[single_fits['p_id'] == participant, 'B'].values[0]
        # print(As_init)

        curr_fitval = np.inf
        possible_starting_points = itertools.product(np.linspace(0, 1, 8), np.linspace(0, 1, 8), np.linspace(0, 1, 8))
        for i in possible_starting_points:
            temp_res = minimize(calc_log_likelihood, x0=np.concatenate(([As_init, Bs_init], i)).tolist(), args=(errors, 'dual state', p_type), bounds=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)), method = 'Nelder-Mead')
            if temp_res.fun < curr_fitval:
                res = temp_res
                curr_fitval = res.fun
        print('participant done: ', participant)
    except:
        print('participant failed: ', participant)
        return participant, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return participant, res.fun, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]


if __name__ == '__main__':
    participant = data['p_id'].unique()
    # participant = [641, 642]
    pool = mp.Pool()
    # single_fit_results = pool.map(fit_single, participant)
    dual_fit_results = pool.map(fit_dual, participant)
    
    # df = pd.DataFrame(single_fit_results, columns =['p_id', 'gof', 'A', 'B', 'Eps'])
    df = pd.DataFrame(dual_fit_results, columns =['p_id', 'gof', 'As', 'Bs', 'Af', 'Bf', 'Eps'])

    df.to_csv('dual_fit_10pcutofferror_results.csv')
    print(df)
