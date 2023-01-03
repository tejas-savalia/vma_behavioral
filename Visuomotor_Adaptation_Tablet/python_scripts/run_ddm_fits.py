import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddm import Model, Sample
from ddm import Model
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICRange
from ddm.functions import fit_adjust_model, display_model
from ddm import Fittable
from ddm.models import LossRobustBIC, LossRobustLikelihood, LossSquaredError
from ddm.functions import fit_adjust_model, fit_model
import seaborn as sns
import pickle
import scipy.io
from multiprocessing import Pool
from ddm import Bound, Drift

def fit_ddm_participant(participant):
    df_errors = pd.read_csv('Curvature_Errors.csv')
    incorrect_thresold = np.array([9/90, 15/90, 20/90, 30/90, 45/90])
    model_fits_4param = np.zeros((12, len(incorrect_thresold)), dtype = object)
    for block in range(12):
        print("Participant: ", participant)
        print("Block: ", block)
        for ic in range(len(incorrect_thresold)):
            df_rt = pd.read_csv('RTs.csv')            
            df_rt['Correct'] = df_errors['Errors'] < incorrect_thresold[ic]            
            df_rt = df_rt[df_rt['Participant_Id'] == participant]
            df_rt = df_rt[df_rt["ITs"] > .001]
            df_rt = df_rt[df_rt["ITs"] < 5]
            df_rt = df_rt.drop(['Trial', 'Unnamed: 0', 'Participant_Id', 'Rotation', 'Emphasis', 'MTs'], axis = 1)

            samp = Sample.from_pandas_dataframe(df_rt[df_rt['Block'] == block], rt_column_name="ITs", correct_column_name="Correct")
            model_fit = Model(name='Simple model (fitted)',
                          drift=DriftConstant(drift=Fittable(minval=-20, maxval=20)),
                          noise=NoiseConstant(noise=Fittable(minval = 0, maxval = 5)),                      
                          bound=BoundConstant(B=Fittable(minval = 0, maxval = 20)),
                          overlay=OverlayNonDecision(nondectime=Fittable(minval = 0, maxval = 1)),
                          dx=.001, dt=.01, T_dur=5)

            try:
                fit_adjust_model(samp, model_fit,
                             fitting_method="differential_evolution",
                             lossfunction=LossRobustLikelihood, verbose=False)
            except:
                print ("In except: ")
                print (participant, block)
            model_fits_4param[block][ic] = model_fit
    return model_fits_4param

class DriftBlock(Drift):
    name = "Drift_per_block"
    required_conditions = ['Block']
    required_parameters = ['VBlock0', 'VBlock1', 'VBlock2', 'VBlock3', 'VBlock4', 'VBlock5',
                           'VBlock6', 'VBlock7', 'VBlock8', 'VBlock9', 'VBlock10', 'VBlock11']
    def get_drift(self, conditions, *args, **kwargs):
        if conditions['Block'] == 0:
            return self.VBlock0
        if conditions['Block'] == 1:
            return self.VBlock1
        if conditions['Block'] == 2:
            return self.VBlock2
        if conditions['Block'] == 3:
            return self.VBlock3
        if conditions['Block'] == 4:
            return self.VBlock4
        if conditions['Block'] == 5:
            return self.VBlock5
        if conditions['Block'] == 6:
            return self.VBlock6
        if conditions['Block'] == 7:
            return self.VBlock7
        if conditions['Block'] == 8:
            return self.VBlock8
        if conditions['Block'] == 9:
            return self.VBlock9
        if conditions['Block'] == 10:
            return self.VBlock10
        if conditions['Block'] == 11:
            return self.VBlock11
class BoundsBlock(Bound):
    name = "Bound_per_block"
    required_conditions = ['Block']
    required_parameters = ['BBlock0', 'BBlock1', 'BBlock2', 'BBlock3', 'BBlock4', 'BBlock5',
                           'BBlock6', 'BBlock7', 'BBlock8', 'BBlock9', 'BBlock10', 'BBlock11']
    def get_bound(self, conditions, *args, **kwargs):
        if conditions['Block'] == 0:
            return self.BBlock0
        if conditions['Block'] == 1:
            return self.BBlock1
        if conditions['Block'] == 2:
            return self.BBlock2
        if conditions['Block'] == 3:
            return self.BBlock3
        if conditions['Block'] == 4:
            return self.BBlock4
        if conditions['Block'] == 5:
            return self.BBlock5
        if conditions['Block'] == 6:
            return self.BBlock6
        if conditions['Block'] == 7:
            return self.BBlock7
        if conditions['Block'] == 8:
            return self.BBlock8
        if conditions['Block'] == 9:
            return self.BBlock9
        if conditions['Block'] == 10:
            return self.BBlock10
        if conditions['Block'] == 11:
            return self.BBlock11

def fit_ddm_2_param_participant(participant):
    #Function to keep Non_dec_time and drift noise constant across blocks.                               
    df_errors = pd.read_csv('Curvature_Errors.csv')
    incorrect_thresold = np.array([9/90, 15/90, 20/90, 30/90, 45/90])
    model_fits_2param = np.zeros((len(incorrect_thresold)), dtype = object)
    print ("Fitting Participant: ", participant)
    for ic in range(len(incorrect_thresold)):
        df_rt = pd.read_csv('RTs.csv')            
        df_rt['Correct'] = df_errors['Errors'] < incorrect_thresold[ic]            
        df_rt = df_rt[df_rt['Participant_Id'] == participant]
        df_rt = df_rt[df_rt["ITs"] > .001]
        df_rt = df_rt[df_rt["ITs"] < 5]
        df_rt = df_rt.drop(['Trial', 'Unnamed: 0', 'Participant_Id', 'Rotation', 'Emphasis', 'MTs'], axis = 1)

        samp = Sample.from_pandas_dataframe(df_rt, rt_column_name="ITs", correct_column_name="Correct")
        model_fit = Model(name='Simple model (fitted)',
                          drift=DriftBlock(VBlock0=Fittable(minval=-20, maxval=20),
                                           VBlock1=Fittable(minval=-20, maxval=20),
                                           VBlock2=Fittable(minval=-20, maxval=20),
                                           VBlock3=Fittable(minval=-20, maxval=20),
                                           VBlock4=Fittable(minval=-20, maxval=20),
                                           VBlock5=Fittable(minval=-20, maxval=20),
                                           VBlock6=Fittable(minval=-20, maxval=20),
                                           VBlock7=Fittable(minval=-20, maxval=20),
                                           VBlock8=Fittable(minval=-20, maxval=20),
                                           VBlock9=Fittable(minval=-20, maxval=20),
                                           VBlock10=Fittable(minval=-20, maxval=20),
                                           VBlock11=Fittable(minval=-20, maxval=20)),

                          bound=BoundsBlock(BBlock0=Fittable(minval=0, maxval=20),
                                           BBlock1=Fittable(minval=0, maxval=20),
                                           BBlock2=Fittable(minval=0, maxval=20),
                                           BBlock3=Fittable(minval=0, maxval=20),
                                           BBlock4=Fittable(minval=0, maxval=20),
                                           BBlock5=Fittable(minval=0, maxval=20),
                                           BBlock6=Fittable(minval=-0, maxval=20),
                                           BBlock7=Fittable(minval=0, maxval=20),
                                           BBlock8=Fittable(minval=0, maxval=20),
                                           BBlock9=Fittable(minval=0, maxval=20),
                                           BBlock10=Fittable(minval=0, maxval=20),
                                           BBlock11=Fittable(minval=0, maxval=20)),

                      noise=NoiseConstant(noise=Fittable(minval = 0, maxval = 5)),                      
                      overlay=OverlayNonDecision(nondectime=Fittable(minval = 0, maxval = 1)),
                      dx=.001, dt=.01, T_dur=5)
        try:
            fit_adjust_model(samp, model_fit,
                         fitting_method="differential_evolution",
                         lossfunction=LossRobustLikelihood, verbose=False)
            print("Participant done: ", participant)
            print("ic: ", ic)
        except:
            print ("In except: ")
            print (participant)
            print (ic)
        model_fits_2param[ic] = model_fit
    return model_fits_2param


def main():
    pool = Pool()
    res = np.zeros(60, dtype = object)
    #res = pool.map(fit_ddm_participant, range(60))
    res = pool.map(fit_ddm_2_param_participant, range(60))
    pickle.dump(res, open('ddm_fits_2param.pickle', 'wb'))

if __name__ == '__main__':
    main()
