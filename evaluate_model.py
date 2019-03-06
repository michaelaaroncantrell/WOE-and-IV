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
import matplotlib.pyplot as plt

from model_performance import *
from rfe import *

def eval_scorecard(supp, X, y, cv_ks = False, max_bin = 5, n_splits = 10, n_bin = 20):
    X = X[supp]
    #For final scorecard, use the whole dataset to train.
    out = run_WOE(X, X, y, y, max_bin = max_bin, keep_all=True)
    X_WOE_tranformed_train, X_WOE_tranformed_test, WOE, WOE_concise = out[0], out[1], out[2], out[3]
    display(WOE_concise.sort_values('IV', ascending = False))
    model = LR(C = 1000000000, penalty = 'l1'
                   , random_state=20181204
              )
    model.fit(X_WOE_tranformed_train, y)
    auc_out = Get_AUC_KS_with_model(model = model
                          , X_train = X_WOE_tranformed_train
                          , X_test = X_WOE_tranformed_train
                          , y_train = y
                          , y_test = y
                          , n_bin = n_bin
                          , verbose = True
#                                     , gain_charts = True
                         )
    display(WOE)
    y_pred = model.predict_proba(X_WOE_tranformed_train)
    y_pred = [y_pred[idx][1] for idx in range(len(y_pred))]

    results_reverse = pd.DataFrame({'y_pred' : [1 - pred for pred in y_pred], 'y' : y })
    gain_chart = compute_gain_chart(results = results_reverse
                                    , num_cat = 20
                                    , bad_name = 'y'
                                    , score_name = 'y_pred')

    credit_results = pd.DataFrame({'credit_score' : X.credit_score
                                  , 'y' : y
                                  })
    credit_gain_chart = compute_gain_chart(results = credit_results
                                          , num_cat = 20
                                          , bad_name = 'y'
                                          , score_name = 'credit_score'
                                          )
    
    plot_ugly_gain_chart(gain_chart)
    plot_pretty_gain_chart(ff_gain_chart = gain_chart
                           , credit_gain_chart = credit_gain_chart
                          )
    
    results = pd.DataFrame({'y_pred': y_pred
                 , 'y' : y
                 })
    #plot model prob vs actual in deciles
    hist_y_lim = plot_vs_historical(results, bad_name = 'y', var_name = 'y_pred')
    # plot credit plots 
    _ = plot_vs_historical(credit_results, bad_name = 'y', var_name = 'credit_score'
                       , var_name_is_pct = False
                      , reverse_var_name = True
                      , ylim = hist_y_lim)
    
    ### y_pred distribution
    pd.DataFrame({'y_pred' : y_pred}).hist()
    plt.show()
    if cv_ks:
        cv_out = Check_model_on_random_splits(X = X
                                              , y = y
                                              , supp = supp
                                              , n_splits = n_splits
                                              , verbose = False
                                              , stats_model = False
#                                              , gain_charts = False
                                             )

        return(model, WOE, auc_out, cv_out, y_pred)
    return(model, WOE, auc_out, y_pred)



############################################################################################
################################# HIDDEN HELPER FUNS #######################################
############################################################################################


def Get_AUC_KS_with_model(model, X_train, X_test, y_train, y_test, n_bin, verbose = True
#                          , gain_charts = False
                         ):
#     model, coef_df = fit_model(X_train, y_train, C, penalty)
    coef_df = pvalues(model, X_train, y_train)
    if verbose:
        display(coef_df)
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
    if verbose:
        display(train_gain_chart)
        display(test_gain_chart)
#     if gain_charts:
#         return(auc_test, auc_train, test_ks, train_ks, test_gain_chart, train_gain_chart)

    return(auc_test, auc_train, test_ks, train_ks, test_gain_chart)

def plot_ugly_gain_chart(gain_chart):
    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    width = 0.4

    bad_num = gain_chart.bad_count.sum()
    gain_chart.bad_count.cumsum().apply(lambda x: x/bad_num).plot(kind='bar', color='red', ax=ax, width=width, position=1)
    gain_chart.count_in_cat.cumsum().apply(lambda x: x/bad_num).plot(kind='bar', color='blue', ax=ax2, width=width, position=0);
    display(display(gain_chart.drop(['good_count', 'acc_good_count', 'acc_good_pct'], axis = 1)))
    return()


def plot_vs_historical(df
                       , bad_name
                       , var_name
                       , ylim = None 
                       , var_name_is_pct = True
                       , reverse_var_name = False):    
    df_bad_mean = df[bad_name].mean()
    df['var_decile'] = pd.qcut(df[var_name], 10)
    bad_rate = df.groupby('var_decile')[bad_name].mean().apply(lambda x: 10 * x/df_bad_mean).reset_index()
    if var_name_is_pct:
        var_mean = df.groupby('var_decile')[var_name].mean().apply(lambda x: round(100*x, 1)).reset_index()
    else:
        var_mean = df.groupby('var_decile')[var_name].mean().apply(lambda x: round(x, 1)).reset_index()
    df_merged = pd.merge(bad_rate, var_mean)
    #bar plot of decile var_name vs bad_name
    if reverse_var_name:
        display(df_merged.sort_values(var_name, ascending = False).plot.bar(x = var_name, y = bad_name))
    else:
        display(df_merged.plot.bar(x = var_name, y = bad_name))
        
    if ylim is not None:
        plt.ylim((0, ylim + 2))

    display(df_merged)
    #line plot
    display(df_merged.plot(x = var_name, y = bad_name))

    #scatter plot with y=x line
    if var_name_is_pct:
        display(df_merged.plot.scatter(x = var_name, y = bad_name))
        plt.plot(x = df_merged[var_name], y = df_merged[bad_name])
        plt.plot(df_merged[var_name], df_merged[var_name], color = 'red')
    if reverse_var_name:
        plt.gca().invert_xaxis()
    plt.show();
    return(df_merged[bad_name].max())
    

def plot_pretty_gain_chart(ff_gain_chart, credit_gain_chart):
    ff_bad_pct = [0] + list(ff_gain_chart.acc_bad_pct) #manually add 0,0 to make graph nice
    ff_good_pct = [0] + list(ff_gain_chart.acc_good_pct)
    
    credit_bad_pct = [0] + list(credit_gain_chart.acc_bad_pct)
    credit_good_pct = [0] + list(credit_gain_chart.acc_good_pct)

    fig = plt.figure() # Create matplotlib figure
    ax = fig.add_subplot(111)
    plt.plot(ff_bad_pct, color = 'blue');
    plt.plot(credit_bad_pct, color = 'red')
    plt.plot(ff_good_pct, color = 'black');
    plt.plot(credit_good_pct, color = 'green')
    ax.legend(["ff acc bad"
               , "credit score acc bad"
               , "ff acc good"
               , "credit score acc good"
              ]
              , loc = 'lower right');
    plt.xticks([])



def Check_model_on_random_splits(X, y, supp, n_splits, verbose, stats_model = True
#                                  , gain_charts = False
                                ):
    
    X_small = X[supp]
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=.5, random_state=1234)
    sss.get_n_splits(X_small, y)
    out_arr = []
    for train_index, test_index in sss.split(X_small, y):
        X_train, X_test = X_small.iloc[train_index], X_small.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        out = run_WOE(X_train, X_test, y_train, y_test, max_bin = 5, keep_all=True)
        X_WOE_tranformed_train, X_WOE_tranformed_test, WOE, WOE_concise = out[0], out[1], out[2], out[3]
        try:
            out = Model_on_vars(supp_var = supp
                      , X_WOE_tranformed_train = X_WOE_tranformed_train
                      , X_WOE_tranformed_test = X_WOE_tranformed_test
                      , y_train = y_train
                      , y_test = y_test
                      , verbose = verbose
                                , stats_model = stats_model
#                                 , gain_charts = gain_charts
                     )
            out_arr.append(out)
        except:
            print('whoops')
    return(out_arr)



def Model_on_vars(supp_var
                  , X_WOE_tranformed_train
                  , X_WOE_tranformed_test
                  , y_train
                  , y_test
                  , C = 10000000
                  , verbose = True
                  , stats_model = True
#                   , gain_charts = False
                 ):
    X_train_rfe = X_WOE_tranformed_train[supp_var]
    X_test_rfe = X_WOE_tranformed_test[supp_var]
    y_train_rfe = y_train
    y_test_rfe = y_test
    
    model = LR(C = C, penalty = 'l1')
    model.fit(X_train_rfe, y_train_rfe)
    
    if stats_model:
        ### COMPARE TO STATS MODEL ###
        logit = sm.Logit(y_train_rfe, X_train_rfe)
        # fit the model
        result = logit.fit()
        print(result.summary2())
        ### END STATS MODEL ###
    
    y_pred = [x[1] for x in list(model.predict_proba(X_test_rfe))]
    if verbose:
        print(roc_auc_score(y_test_rfe, y_pred))
    
    return(Get_AUC_KS_with_model(model = model
                          , X_train = X_train_rfe
                          , X_test = X_test_rfe
                          , y_train = y_train_rfe
                          , y_test = y_test_rfe
                          , n_bin = 20
                          , verbose = verbose
#                                  , gain_charts = gain_charts
                         ))