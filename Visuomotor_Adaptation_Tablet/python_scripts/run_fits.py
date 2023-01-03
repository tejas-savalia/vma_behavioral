#%%
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
from all_models import *
#from transfer_models import *
import sys

# %%
def main(num_fit_trials):
    curvatures_smooth = pickle.load(open('curvatures_smooth.pickle', 'rb'))
    curvatures_smooth = curvatures_smooth/90
    #curvatures_smooth = pickle.load(open('mean_drawn_grouperrors.pickle', 'rb'))
    #curvatures_smooth = curvatures_smooth/90

    #curvatures_smooth = pickle.load(open('auc_smooth.pickle', 'rb'))
    #curvatures_smooth = curvatures_smooth/np.nanmax(curvatures_smooth)    
    
    #print ("AUC Curvatures Loaded. In Fit routine")
    #print (num_fit_trials)

    #%% Parallel run and dump fits
    
    #fits = run_fits_single(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]))
    #with open('fit_single_CV_704_auc.pickle', 'wb') as f:
    #    pickle.dump(fits, f)
    #f.close()
    
    #fits = run_fits_dual(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]))
    #with open('fit_dual_CV_704_auc.pickle', 'wb') as f:
    #    pickle.dump(fits, f)
    #f.close()

    #curvatures_smooth = pickle.load(open('mad_smooth.pickle', 'rb'))
    #curvatures_smooth = curvatures_smooth/np.nanmax(curvatures_smooth)


    #print ("MAD Curvatures Loaded. In Fit routine")
    #print (num_fit_trials)

    #%% Parallel run and dump fits
    
    #fits = run_fits_single(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]))
    #with open('fit_single_CV_704_mad.pickle', 'wb') as f:
    #    pickle.dump(fits, f)
    #f.close()
    
    #fits = run_fits_dual(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]))
    #with open('fit_dual_CV_704_mad.pickle', 'wb') as f:
    #    pickle.dump(fits, f)
    #f.close()
    #curvatures_smooth = pickle.load(open('avg_smooth.pickle', 'rb'))
    #curvatures_smooth = curvatures_smooth/np.nanmax(curvatures_smooth)    
    
    #print ("AVG Curvatures Loaded. In Fit routine")
    #print (num_fit_trials)
    
    #fits = run_fits_dual(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]))
    #with open('fit_dual_CV_704_avg.pickle', 'wb') as f:
    #    pickle.dump(fits, f)
    #f.close()

    #fits = run_fits_single(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]))
    #with open('fit_single_CV_704_avg.pickle', 'wb') as f:
    #    pickle.dump(fits, f)
    #f.close()
    
    #curvatures_smooth = pickle.load(open('ide_smooth.pickle', 'rb'))
    #curvatures_smooth = curvatures_smooth/np.nanmax(curvatures_smooth)    
    
    #print ("IDE Curvatures Loaded. In Fit routine")
    #print (num_fit_trials)

    #%% Parallel run and dump fits
    
    #fits = run_fits_single(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]))
    #with open('fit_single_CV_704_ide.pickle', 'wb') as f:
    #    pickle.dump(fits, f)
    #f.close()
    
    #fits = run_fits_dual(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]))
    #with open('fit_dual_CV_704_ide.pickle', 'wb') as f:
    #    pickle.dump(fits, f)
    #f.close()

    
    fits = run_fits_dual(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]), int(num_fit_trials[3]))
    with open('fit_dual_CV_640_bestfit_starting_point.pickle', 'wb') as f:
        pickle.dump(fits, f)
    f.close()
    fits = run_fits_single(curvatures_smooth, int(num_fit_trials[1]), int(num_fit_trials[2]), int(num_fit_trials[3]))
    with open('fit_single_CV_640_bestfit_starting_point.pickle', 'wb') as f:
        pickle.dump(fits, f)
    f.close()



if __name__ == '__main__':
    main(sys.argv)
