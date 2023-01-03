import hddm
import pickle
import pandas as pd
import numpy as np
df_rt = hddm.load_csv('RTs.csv')
df_rt = df_rt.drop(['Unnamed: 0', 'MTs'], axis = 1)
df_error = pd.read_csv(open('Angular_errors.csv'))

df_rt['response'] = 0
df_rt['response'][df_error['Error'] < np.pi/9] = 1
df_rt['response'][df_error['Error'] > np.pi/9] = 0

df_rt = df_rt.rename(columns={'participant_id': 'subj_idx', 'ITs':'rt'})

m_stim = hddm.HDDM(df_rt, depends_on={'v': ['block', 'condition'], 'a':['block', 'condition']})
m_stim.find_starting_values()
m_stim.sample(10000, burn=1000, dbname = 'hddm_traces.db', db = 'pickle')
m_stim.save('hddm_fit')

