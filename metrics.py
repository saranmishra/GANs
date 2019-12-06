#This is will be the page for the metrics

import numpy as np
import sklearn 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_arrays
from math import sqrt

#NEEDS WORK

class MAE:
    
    sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
    

class RMSE:
    rms = sqrt(mean_squared_error(y_actual, y_predicted))

class MAPE:
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = check_arrays(y_true, y_pred)

        ## Note: does not handle mix 1d representation
        #if _is_1d(y_true): 
        #    y_true, y_pred = _check_1d_array(y_true, y_pred)

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class AR:
    avg = float(sum(list))/len(list)
    
