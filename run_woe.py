from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_selection import RFE
# from scipy import stats
from scipy.stats import chi2
from copy import deepcopy
import random

from woe import data_vars
##########################
### MAIN WOE FUNCTIONS ###
##########################
def run_WOE(X_train, X_test, y_train, y_test, max_bin = 5):
    ''' 
    Computes WOE on X_train, X_test. If keep_all, then WOE-transforms train and test and
    returns. Otherwise, subsets variables on those with IV > 0.02, and looks at
    variables with abs(correlation) > 0.5 and drops the variable with lower IV.
    WOE-transforms remaining variables and returns.
    '''
    
    #Get WOE
    WOE, WOE_concise = data_vars(X_train, y_train, max_bin = max_bin) 
    
    #TRANSFORM BOTH TRAIN/TEST WITH THIS OUTPUT
    X_WOE_tranformed_train = woe_transform_all_cols(X_WOE_subset_train, WOE)
    X_WOE_tranformed_test = woe_transform_all_cols(X_WOE_subset_test, WOE)
    
    return(X_WOE_tranformed_train, X_WOE_tranformed_test, WOE, WOE_concise)



############################################################################################
################################# HIDDEN HELPER FUNS #######################################
############################################################################################


def woe_transform_all_cols(X, WOE_df):
    for col in list(X):
        X = woe_transform_for_arb_col(col, X, WOE_df)
    return(X)


def woe_transform_for_arb_col(col, X, WOE_df):
    curr_WOE = WOE_df[WOE_df.VAR_NAME == col]
    def tmp_fun(x, curr_WOE):
#         if X[col].dtype == 'object':
#             print(col)
#             display(curr_WOE)
        #if x is null, find row with min = max = null and highest counts.
#         if isinstance(x, float):
#             if np.isnan(x):
# #             print('NAN!')
#                 curr_WOE_null = curr_WOE[curr_WOE.MIN_VALUE.isna()]
# #             print(curr_WOE_null[curr_WOE_null.COUNT == curr_WOE_null.COUNT.max()].WOE)
#                 return(float(curr_WOE_null[curr_WOE_null.COUNT == curr_WOE_null.COUNT.max()].WOE))
            
        boo_arr = (curr_WOE.MIN_VALUE <= x) & (x <= curr_WOE.MAX_VALUE)
        #this is to handle edge case when max value of test is greater than that of train
        if not any(boo_arr): 
            if x < curr_WOE.iloc[0].MIN_VALUE:
                WOE_bin_idx = 0
            elif x > curr_WOE.iloc[len(curr_WOE) - 1].MAX_VALUE:
                WOE_bin_idx = len(curr_WOE)
#             else:
#                 print(col, x)
        if len([i for i, x in enumerate(boo_arr) if x]) == 0:
#             print('assigning value to 0', x, col)
            return(0) #a hack because x may be inbetween row 1 max and row 2 min...
                #unfortunately nans aren't handled well and are getting mapped to 0 too
                #could fix this with more time. doesn't seem to be a big issue

        WOE_bin_idx = [i for i, x in enumerate(boo_arr) if x][0]
        return(curr_WOE.iloc[WOE_bin_idx].WOE)

    X_new = X.copy()
    # WARNING !!! FILLING NAS BECAUSE OF SHITTY NA HANDLING IN WOE SOFTWARE #
    X_new[col] = X_new[col].apply(lambda x: tmp_fun(x, curr_WOE)).fillna(0)
    return(X_new)