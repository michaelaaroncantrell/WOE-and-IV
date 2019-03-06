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
from run_woe import run_WOE

from model_performance import *

### RFE / ITERATE WOE ### -- TODO: fix
def Iterate_RFE(X, y
                , n_splits = 3, n_bin_KS = 20, n_features_to_select = 8
                , random_state = 0, test_size = 0.3, max_bin_WOE = 5):

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    sss.get_n_splits(X, y)
    out_arr = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print('X_train size', X_train.shape, 'X_test size', X_test.shape)
        
        out = run_WOE(X_train, X_test, y_train, y_test, max_bin = max_bin_WOE)
        X_WOE_tranformed_train, X_WOE_tranformed_test, WOE, WOE_concise = out[0], out[1], out[2], out[3]
        display(WOE_concise.sort_values('IV', ascending = False))
        
        #NO SCALING
        C = 0.075
        # C = 1
        estimator = LR(C = C, penalty = 'l1')
        selector = RFE(estimator, n_features_to_select = n_features_to_select, step=1)
        selector = selector.fit(X_WOE_tranformed_train, y_train)
        supp_vars = np.array(list(X_WOE_tranformed_train))[np.array(selector.support_)]
        print(supp_vars)
        
        # X_train_rfe = X_WOE_tranformed_train[supp_vars]
        # X_test_rfe = X_WOE_tranformed_test[supp_vars]
        # y_train_rfe = y_train
        # y_test_rfe = y_test
        # # estimator.fit(X_train_rfe, y_train_rfe)
        # y_pred = [x[1] for x in list(selector.predict_proba(X_WOE_tranformed_test))]
        # print(roc_auc_score(y_test_rfe, y_pred))

        # Get_AUC_KS_with_train_test(X_train = X_train_rfe
        #                            , X_test = X_test_rfe
        #                            , y_train = y_train
        #                            , y_test = y_test
        #                            , C = C
        #                            , penalty = 'l1'
        #                            , n_bin = n_bin_KS
        #                           )
        
        out_arr.append([X_train
                       , X_test
                       , y_train
                       , y_test
                       , X_WOE_tranformed_train
                       , X_WOE_tranformed_test
                       , WOE
                       , WOE_concise
                      , supp_vars
                      , selector])
    return(out_arr)


def Examine_Iterate_RFE_results(out):
    all_cols = sorted(list(set(out[0][-2]).union(set(out[1][-2])).union(set(out[2][-2]))))
    counts = []
    for col in all_cols:
        n = 0
        if col in out[0][-2]:
            n += 1
        if col in out[1][-2]:
            n += 1
        if col in out[2][-2]:
            n += 1
        counts.append([col, n])
    counts.sort(key=lambda tup: tup[1]) 
    return(counts)


############################################################################################
################################# HIDDEN HELPER FUNS #######################################
############################################################################################

def Get_AUC_KS_with_train_test(X_train, X_test, y_train, y_test, C, penalty, n_bin):
    model, coef_df = fit_model(X_train, y_train, C, penalty)
    auc_test, auc_train, test_ks, train_ks, test_gain_chart, train_gain_chart, results = evaluate_model(X_test
                                                                                   , y_test
                                                                                   , X_train
                                                                                   , y_train
                                                                                   , model
                                                                                   , coef_df
                                                                                                       , n_bin = n_bin)
    print('test auc: {}\ntrain auc: {}'.format(auc_test, auc_train))
    print('test ks: {}\ntrain ks: {}'.format(test_ks, train_ks))
#     print('train auc: {}\ntrain ks: {}'.format(train_auc, train_ks))
    display(train_gain_chart)
    display(test_gain_chart)
    
    
def evaluate_model(X_test, y_test, X_train, y_train, model, coef_df, n_bin):
    auc_test=roc_auc_score(y_test, [x[1] for x in list(model.predict_proba(X_test))])
    auc_train=roc_auc_score(y_train, [x[1] for x in list(model.predict_proba(X_train))])
    results=pd.DataFrame()
    results['bad']= y_test
    #    results['prob'] = model.pred_proba()
    results['pred']=coef_df[coef_df.variable=='intercept'].coefficients[0]
    for v in list(X_test):
        results['pred']=results['pred']+list(coef_df[coef_df.variable==v].coefficients)[0]*X_test[v]
    results['prob']=[x[0] for x in list(model.predict_proba(X_test))]
    results['score']=500-(20/np.log(2))*(np.log(15)+results.pred)
    test_gain_chart=compute_gain_chart(results, n_bin)
#     gain_chart=compute_gain_chart(results,5)
    test_ks=test_gain_chart.ks.max()
    
    train_results=pd.DataFrame()
    train_results['bad']= y_train
    #    results['prob'] = model.pred_proba()
    train_results['pred']=coef_df[coef_df.variable=='intercept'].coefficients[0]
    for v in list(X_train):
        train_results['pred']=train_results['pred']+list(coef_df[coef_df.variable==v].coefficients)[0]*X_train[v]
    train_results['prob']=[x[0] for x in list(model.predict_proba(X_train))]
    train_results['score']=500-(20/np.log(2))*(np.log(15)+train_results.pred)
    train_gain_chart=compute_gain_chart(train_results,n_bin)
#     gain_chart=compute_gain_chart(results,5)
    train_ks=train_gain_chart.ks.max()
    return auc_test, auc_train, test_ks, train_ks, test_gain_chart, train_gain_chart, results


def fit_model(X_train, y_train, C, penalty):
    model=LR(penalty=penalty,C=C)
    model.fit(X_train, y_train)
    coef_df=pvalues(model, X_train, y_train)
    return model, coef_df

