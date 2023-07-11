import numpy as np
import pandas as pd
import scipy.stats as stat
from scipy.optimize import minimize, basinhopping
import itertools
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")


def single_state_model(A, B, num_trials, p_type):
    rotation_estimate = np.zeros(num_trials)
    error = np.zeros(num_trials)
    if p_type == 'Sudden':
        # rotation = np.pi/3
        for trial in range(1, num_trials):
            if trial in range(64*7 - 1, 64*8 - 1):
            # if trial > 64*7 - 1 and trial <  64*8 - 1:

                rotation = -np.pi/3
            else:
                rotation = np.pi/3

            # error[trial-1] = np.abs(rotation - rotation_estimate[trial-1])
            error[trial-1] = rotation - rotation_estimate[trial-1]

            rotation_estimate[trial] = A*rotation_estimate[trial-1] + B*error[trial-1]
    else:
        rotation = np.pi/18
        for trial in range(1, num_trials):
            # error[trial-1] = np.abs(rotation - rotation_estimate[trial-1])
            error[trial-1] = rotation - rotation_estimate[trial-1]

            rotation_estimate[trial] = A*rotation_estimate[trial-1] + B*error[trial-1]
            if trial%64 == 0:
                if rotation < np.pi/3:
                    rotation = rotation + np.pi/18
                else:
                    rotation = np.pi/3

            if trial in range(64*7 - 1, 64*8 - 1):
            # if trial > 64*7 - 1 and trial <  64*8 - 1:

                rotation = -np.pi/3

    # error[trial] = np.abs(rotation - rotation_estimate[trial])
    error[trial-1] = rotation - rotation_estimate[trial-1]

    return error

def dual_state_model(As, Bs, Af, Bf, num_trials, p_type):
    rotation_estimate = np.zeros(num_trials)
    fast_estimate = np.zeros(num_trials)
    slow_estimate = np.zeros(num_trials)
    
    error = np.zeros(num_trials)
    if p_type == 'Sudden':
        # rotation = np.pi/3

        for trial in range(1, num_trials):

            if trial > 64*7 - 1 and trial < 64*8:
                rotation = -np.pi/3
            else:
                rotation = np.pi/3

            # error[trial-1] = np.abs(rotation - rotation_estimate[trial-1])
            error[trial-1] = rotation - rotation_estimate[trial-1]

            fast_estimate[trial] = Af*fast_estimate[trial-1] + Bf*error[trial-1]
            slow_estimate[trial] = As*slow_estimate[trial-1] + Bs*error[trial-1]
            rotation_estimate[trial] = fast_estimate[trial] + slow_estimate[trial]
            
    else:
        rotation = np.pi/18
        for trial in range(1, num_trials):


            # error[trial-1] = np.abs(rotation - rotation_estimate[trial-1])
            error[trial-1] = rotation - rotation_estimate[trial-1]

            fast_estimate[trial] = Af*fast_estimate[trial-1] + Bf*error[trial-1]
            slow_estimate[trial] = As*slow_estimate[trial-1] + Bs*error[trial-1]
            rotation_estimate[trial] = fast_estimate[trial] + slow_estimate[trial]

            if trial%64 == 0:
                if rotation < np.pi/3:
                    rotation = rotation + np.pi/18
                else:
                    rotation = np.pi/3

            if trial > 64*7 - 1 and trial < 64*8:
                rotation = -np.pi/3

    # error[trial] = np.abs(rotation - rotation_estimate[trial-1])
    error[trial-1] = rotation - rotation_estimate[trial-1]

    return error

def calc_log_likelihood(params, data, model, p_type, fit_type = 'regular', train_indices = None):
    if model == 'single state':
        # if any(params[:-1]) < 0 or any(params) > 1:
        #     return 100000
        model_pred = single_state_model(params[0], params[1], len(data), p_type)
    else:
        if params[0] < params[2]  or params[1] > params[3]:
            return 100000        
        model_pred = dual_state_model(params[0], params[1], params[2], params[3], len(data), p_type)

    if fit_type == 'cv':
        train_data = data[train_indices]
        train_model_pred = model_pred[train_indices]
        log_lik = np.mean(stat.norm.logpdf(train_data, train_model_pred, params[-1]))
    else:
        log_lik = np.mean(stat.norm.logpdf(data, model_pred, params[-1]))
    # print(params)
    return -log_lik


############ Fitting Functions
#Load Data
data = pd.read_csv('df_allphases.csv')
data = data.loc[data['block'] >= 1].reset_index().drop('index', axis = 1)


def fit_single(participant):
    print('participant started: ', participant)
    try:
        errors = data.loc[data['p_id'] == participant, 'init signed errors'].values
        print(len(errors))

        p_type = data.loc[data['p_id'] == participant, 'Rotation'].unique()
        curr_fitval = np.inf
        possible_starting_points = itertools.product(np.linspace(0.0001, 0.9999, 8), np.linspace(0.0001, 0.9999, 8), np.linspace(0.01, 5, 8))
        for i in possible_starting_points:
            temp_res = minimize(calc_log_likelihood, x0=i, args=(errors, 'single state', p_type), bounds=((.0001, .9999), (.0001, .9999), (0.01, 5)), method = 'Nelder-Mead')
            if temp_res.fun < curr_fitval:
                res = temp_res
                curr_fitval = res.fun
        print('participant done: ', participant, res.fun, res.x)
    except:
        print('participant failed: ', participant)
        return participant, np.nan, np.nan, np.nan, np.nan
    return participant, res.fun, res.x[0], res.x[1], res.x[2]

#load single fits to use as slow learning starting points.

def fit_dual(participant):
    single_fits = pd.read_csv('model_results/single_fit_initsignederror_results.csv')
    print('participant started: ', participant)

    # try:
    errors = data.loc[data['p_id'] == participant, 'init signed errors'].values
    p_type = data.loc[data['p_id'] == participant, 'Rotation'].unique()
    As_init = single_fits.loc[single_fits['p_id'] == participant, 'A'].values[0]
    Bs_init = single_fits.loc[single_fits['p_id'] == participant, 'B'].values[0]
    Eps_init = single_fits.loc[single_fits['p_id'] == participant, 'Eps'].values[0]

    # print(As_init)

    curr_fitval = np.inf
    possible_starting_points = itertools.product(np.linspace(0.0001, 0.9999, 16), np.linspace(0.0001, 0.9999, 16))
    # possible_starting_points = itertools.product(np.linspace(0, 1, 4), np.linspace(0, 1, 4), np.linspace(0, 1, 4), np.linspace(0, 1, 4), np.linspace(0.01, 5, 4))

    for i in possible_starting_points:
        x0=np.concatenate(([As_init, Bs_init], i, [Eps_init])).tolist()
        # x0 = i
        temp_res = minimize(calc_log_likelihood, x0=x0, args=(errors, 'dual state', p_type), method = 'L-BFGS-B', bounds = ((0.0001, .9999), (0.0001, .9999), (0.0001, .9999), (0.0001, .9999), (0.01, 5)))
        if temp_res.fun < curr_fitval:
            res = temp_res
            curr_fitval = res.fun
        # minimizer_kwargs = {'method':'Nelder-Mead', "args":(errors, 'dual state', p_type), 'bounds' : ((0, 1), (0, 1), (0, As_init), (Bs_init, 1), (0.01, 5))}
        # res = basinhopping(calc_log_likelihood, x0=res.x, minimizer_kwargs=minimizer_kwargs)
        # if  curr_fitval >= 10000:
        #     temp_res = minimize(calc_log_likelihood, x0=[0.5, 0.5, 0.4, 0.6, 0.5], args=(errors, 'dual state', p_type), bounds=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)), method = 'Nelder-Mead')

        #     res = temp_res
        #     curr_fitval = res.fun

    print('participant done: ', participant, res.fun, res.x)
# except:
    # print('participant failed: ', participant)
    # return participant, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return participant, res.fun, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]



def fit_single_cv(participant, errors, p_type, train_indices, test_indices):
    # print('participant started: ', participant)
    single_fits = pd.read_csv('model_results/single_fit_initsignederror_results.csv')

    starting_point = single_fits.loc[single_fits['p_id'] == participant, ['A', 'B', 'Eps']].values.tolist()       
    res = minimize(calc_log_likelihood, x0=starting_point, args=(errors, 'single state', p_type, 'cv', train_indices), bounds=((0, 1), (0, 1), (0.01, 5)), method = 'Nelder-Mead')
    test_gof = calc_log_likelihood(res.x, errors, 'single state', p_type, 'cv', test_indices)
    # print('participant done: ', participant)
    # except:
    #     print('participant failed: ', participant)
    #     return participant, np.nan, np.nan, np.nan, np.nan, np.nan
    return [participant, res.fun, test_gof, res.x[0], res.x[1], res.x[2]]

#load single fits to use as slow learning starting points.

def fit_dual_cv(participant, errors, p_type, train_indices, test_indices):
    # print('participant started: ', participant)
    dual_fits = pd.read_csv('model_results/dual_fit_initsignederror_results.csv')

    # try:
    starting_point = dual_fits.loc[dual_fits['p_id'] == participant, ['As', 'Bs', 'Af', 'Bf', 'Eps']].values.tolist()       
    res = minimize(calc_log_likelihood, x0=starting_point, args=(errors, 'dual state', p_type, 'cv', train_indices), bounds=((0, 1), (0, 1), (0, 1), (0, 1), (0.01, 5)), method = 'Nelder-Mead')
    test_gof = calc_log_likelihood(res.x, errors, 'dual state', p_type, 'cv', test_indices)
    # print('participant done: ', participant)
    # except:
    #     print('participant failed: ', participant)
        # return participant, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return [participant, res.fun, test_gof, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]]

def fit_cv(participant):
    errors = data.loc[data['p_id'] == participant, 'init signed errors'].values
    p_type = data.loc[data['p_id'] == participant, 'Rotation'].unique()
    train_indices = np.sort(np.random.choice(np.arange(len(errors)), int(0.9*len(errors)), replace = False))
    test_indices = np.sort(np.delete(np.arange(len(errors)), train_indices)) 
    single_fit_res = fit_single_cv(participant, errors, p_type, train_indices, test_indices)
    dual_fit_res = fit_dual_cv(participant, errors, p_type, train_indices, test_indices)
    return np.concatenate([single_fit_res, dual_fit_res])


if __name__ == '__main__':
    participant = data['p_id'].unique()
    # participant = [641, 642]
    pool = mp.Pool()
    single_fit_results = pool.map(fit_single, participant)
    df = pd.DataFrame(single_fit_results, columns =['p_id', 'gof', 'A', 'B', 'Eps'])
    df.to_csv('model_results/single_fit_initerror_results.csv')

    dual_fit_results = pool.map(fit_dual, participant)
    df = pd.DataFrame(dual_fit_results, columns =['p_id', 'gof', 'As', 'Bs', 'Af', 'Bf', 'Eps'])
    df.to_csv('model_results/dual_fit_initsignederror_results.csv')

    single_fit_df = []
    dual_fit_df = []
    for i in range(100):
        # print(pool.map(fit_cv, participant))
        res_ = np.array(pool.map(fit_cv, participant))
        # print(res_)
        # print(single_fit_results)
        single_fit_res_df = pd.DataFrame(res_[:, :6], columns =['p_id', 'gof', 'test gof', 'A', 'B', 'Eps'])
        dual_fit_res_df = pd.DataFrame(res_[:, 6:], columns =['p_id', 'gof', 'test gof', 'As', 'Bs', 'Af', 'Bf', 'Eps'])

        single_fit_res_df['cv itr'] = i
        single_fit_df.append(single_fit_res_df)
        dual_fit_res_df['cv itr'] = i
        dual_fit_df.append(dual_fit_res_df)
        print('cv iteration done: ', i)

    df_full_single = pd.concat(single_fit_df)
    df_full_single.to_csv('model_results/single_fit_initsignederror_results_cv.csv', index = False)

    df_full_dual = pd.concat(dual_fit_df)
    df_full_dual.to_csv('model_results/dual_fit_initsignederror_results_cv.csv', index=False)


    # dual_fit_results = pool.map(fit_dual, participant)
    # df = pd.DataFrame(dual_fit_results, columns =['p_id', 'gof', 'test gof', 'As', 'Bs', 'Af', 'Bf', 'Eps'])
    # df.to_csv('model_results/dual_fit_initerror_results_cv.csv')
    # df = []
    # for i in range(100):
    #     dual_fit_results = pool.map(fit_dual_cv, participant)
    #     temp_df = pd.DataFrame(dual_fit_results, columns =['p_id', 'gof', 'test gof', 'As', 'Bs', 'Af', 'Bf', 'Eps'])
    #     temp_df['cv itr'] = i
    #     df.append(temp_df)
    #     print('cv iteration done: ', i)
    # df_full = pd.concat(df)
    # df_full.to_csv('model_results/dual_fit_avgerror_results_cv.csv')


    # print(df)

