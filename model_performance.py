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


# def make_report(y_pred, y_true):



def pvalues(model, X, y):
    params = np.append(model.intercept_,model.coef_)
#     predictions = [x for x in list(model.predict(X))]
    predictions = [x[0] for x in list(model.predict_proba(X))]

    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    se_b = np.sqrt(np.sqrt(var_b))

    #ts_b = params/ sd_b
    chi_square_b=((params/se_b)**2)

    #p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
    p_values=[1 - chi2.cdf(i, 1) for i in chi_square_b]

    se_b = np.round(se_b,3)
    #ts_b = np.round(ts_b,3)
    chi_square_b=np.round(chi_square_b,4)
    p_values = np.round(p_values,decimals=4)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()
    myDF3['variable'] = ['intercept']+list(X)
    #myDF3["coefficients"],myDF3["standard errors"],myDF3["t values"],myDF3["p values"] = [params,sd_b,ts_b,p_values]
    myDF3["coefficients"],myDF3["standard_errors"],myDF3["wald_chi-square"],myDF3["p_values"] = [params,se_b,chi_square_b,p_values]
    return myDF3


def compute_gain_chart(results, num_cat, bad_name = 'bad', score_name = 'score'):
    results=results.sort_values(score_name).reset_index(drop=True)
    count_cat=len(results)//num_cat
    gain_chart=pd.DataFrame()
    gain_chart['count_in_cat']=[count_cat for i in range(num_cat-1)]+[len(results)-count_cat*(num_cat-1)]
    gain_chart['LB_score']=[results.iloc[i*count_cat][score_name] for i in range(num_cat)]
    gain_chart['UB_score']=[results.iloc[(i+1)*count_cat-1][score_name] for i in range(num_cat-1)]+[results.iloc[-1][score_name]]
    gain_chart['bad_count']=[results.iloc[i*count_cat:(i+1)*count_cat][bad_name].sum() for i in range(num_cat-1)] + [results.iloc[(num_cat-1)*count_cat:][bad_name].sum()]
    gain_chart['good_count']=gain_chart.count_in_cat-gain_chart.bad_count
    gain_chart['acc_bad_count']=gain_chart.bad_count.cumsum()
    gain_chart['acc_good_count']=gain_chart.good_count.cumsum()
    gain_chart['acc_bad_pct']=gain_chart['acc_bad_count']/np.sum(gain_chart['bad_count'])
    gain_chart['acc_good_pct']=gain_chart['acc_good_count']/np.sum(gain_chart['good_count'])
    gain_chart['ks']=gain_chart['acc_bad_pct']-gain_chart['acc_good_pct']
    return gain_chart


def plot_credit_score(df, bins):
    tmp = df[['credit_score', 'ever_60']].copy()
    tmp['bin'] = pd.cut(df.credit_score, bins = bins)
    means = tmp.groupby('bin').ever_60.mean()
    means.plot.bar();
    means = means.reset_index().rename(columns = {'ever_60' : 'bad pct'})
    counts = tmp.groupby('bin').ever_60.count().reset_index().rename(columns = {'ever_60' : 'counts'})
    return(pd.merge(means,counts))


def plot_default_by_score(df, bins, credit_var, target_var):
    tmp = df[[credit_var, target_var]].copy()
    tmp['bin'] = pd.cut(df[credit_var], bins = bins)
    means = tmp.groupby('bin')[target_var].mean()
    means.plot.bar();
    means = means.reset_index().rename(columns = {target_var : 'bad pct'})
    counts = tmp.groupby('bin')[target_var].count().reset_index().rename(columns = {target_var : 'counts'})
    return(pd.merge(means,counts))