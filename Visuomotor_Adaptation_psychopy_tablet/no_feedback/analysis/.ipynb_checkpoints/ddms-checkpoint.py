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
from ddm import Bound
from ddm import Drift
from multiprocessing import Pool
import pickle

class BoundBlocks(Bound):
    name = "constant"
    required_parameters = ["BBlock0", "BBlock1", "BBlock2", "BBlock3", "BBlock4", "BBlock5",
                          "BBlock6", "BBlock7", "BBlock8", "BBlock9", "BBlock10", "BBlock11"]
    required_conditions = ['block']
    def get_bound(self, conditions, *args, **kwargs):
        if conditions['block'] == 0:
            return self.BBlock0
        if conditions['block'] == 1:
            return self.BBlock1
        if conditions['block'] == 2:
            return self.BBlock2
        if conditions['block'] == 3:
            return self.BBlock3
        if conditions['block'] == 4:
            return self.BBlock4
        if conditions['block'] == 5:
            return self.BBlock5
        if conditions['block'] == 6:
            return self.BBlock6
        if conditions['block'] == 7:
            return self.BBlock7
        if conditions['block'] == 8:
            return self.BBlock8
        if conditions['block'] == 9:
            return self.BBlock9
        if conditions['block'] == 10:
            return self.BBlock10
        if conditions['block'] == 11:
            return self.BBlock11

class DriftBlocks(Drift):
    name = "constant"
    required_parameters = ["VBlock0", "VBlock1", "VBlock2", "VBlock3", "VBlock4", "VBlock5",
                          "VBlock6", "VBlock7", "VBlock8", "VBlock9", "VBlock10", "VBlock11"]
    required_conditions = ['block']

    def get_drift(self, conditions, *args, **kwargs):
        if conditions['block'] == 0:
            return self.VBlock0
        if conditions['block'] == 1:
            return self.VBlock1
        if conditions['block'] == 2:
            return self.VBlock2
        if conditions['block'] == 3:
            return self.VBlock3
        if conditions['block'] == 4:
            return self.VBlock4
        if conditions['block'] == 5:
            return self.VBlock5
        if conditions['block'] == 6:
            return self.VBlock6
        if conditions['block'] == 7:
            return self.VBlock7
        if conditions['block'] == 8:
            return self.VBlock8
        if conditions['block'] == 9:
            return self.VBlock9
        if conditions['block'] == 10:
            return self.VBlock10
        if conditions['block'] == 11:
            return self.VBlock11

def run_ddm_fits(participant):
    print(participant)
    df_rt = pd.read_csv(open('RTs.csv'))
    df_error = pd.read_csv(open('Angular_errors.csv'))
    df_rt['Correct'] = 0
    df_rt['Correct'][df_error['Error'] < np.pi/9] = 1
    df_rt['Correct'][df_error['Error'] > np.pi/9] = 0

    df_rt = df_rt[df_rt['participant_id'] == participant]

    df_rt = df_rt[df_rt["ITs"] > .001]
    df_rt = df_rt[df_rt["ITs"] < 5]
    df_rt = df_rt.drop(['Trial', 'Unnamed: 0', 'participant_id', 'condition', 'MTs'], axis = 1)

    samp = Sample.from_pandas_dataframe(df_rt, rt_column_name="ITs", correct_column_name="Correct")
    try:
        model_fits = fit_model(samp, 
                               bound=BoundBlocks(BBlock0 = Fittable(minval = 0, maxval = 20),
                                                 BBlock1 = Fittable(minval = 0, maxval = 20),
                                                 BBlock2 = Fittable(minval = 0, maxval = 20),
                                                 BBlock3 = Fittable(minval = 0, maxval = 20),
                                                 BBlock4 = Fittable(minval = 0, maxval = 20),
                                                 BBlock5 = Fittable(minval = 0, maxval = 20),
                                                 BBlock6 = Fittable(minval = 0, maxval = 20),
                                                 BBlock7 = Fittable(minval = 0, maxval = 20),
                                                 BBlock8 = Fittable(minval = 0, maxval = 20),
                                                 BBlock9 = Fittable(minval = 0, maxval = 20),
                                                 BBlock10 = Fittable(minval = 0, maxval = 20),
                                                 BBlock11 = Fittable(minval = 0, maxval = 20)),
                               drift=DriftBlocks(VBlock0 = Fittable(minval = -20, maxval = 20),
                                                 VBlock1 = Fittable(minval = -20, maxval = 20),
                                                 VBlock2 = Fittable(minval = -20, maxval = 20),
                                                 VBlock3 = Fittable(minval = -20, maxval = 20),
                                                 VBlock4 = Fittable(minval = -20, maxval = 20),
                                                 VBlock5 = Fittable(minval = -20, maxval = 20),
                                                 VBlock6 = Fittable(minval = -20, maxval = 20),
                                                 VBlock7 = Fittable(minval = -20, maxval = 20),
                                                 VBlock8 = Fittable(minval = -20, maxval = 20),
                                                 VBlock9 = Fittable(minval = -20, maxval = 20),
                                                 VBlock10 = Fittable(minval = -20, maxval = 20),
                                                 VBlock11 = Fittable(minval = -20, maxval = 20)),
                               overlay=OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.5)),

                               noise=NoiseConstant(noise=Fittable(minval=0.01, maxval=5)),

                               lossfunction=LossRobustLikelihood, 
                               verbose = False
                              )
        print("Participant done: ", participant)
    except:
        print("participant failed: ", participant)
        model_fits = 0
    return model_fits
    
def main():
        pool = Pool()
        model_fits = np.array(pool.map(run_ddm_fits, range(14)))
        pickle.dump(model_fits, open('pyddm_fits.pickle', 'wb'))


if __name__ == '__main__':
    main()
