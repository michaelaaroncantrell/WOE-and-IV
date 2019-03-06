#bins may equal a large number
#this function can be looped over
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
import matplotlib.pyplot

def summarize_var(X_cop, col, bins):
    X_cop[col].hist(bins = bins);
    matplotlib.pyplot.show()
    means = X_cop.groupby(col).ever_60.mean().reset_index().rename(columns = {'ever_60' : 'bad pct'})
    X_cop.groupby(col).ever_60.mean().reset_index().plot.bar(x = col, y = 'ever_60')
#     X_cop.groupby(col).ever_60.mean().plot(x = '').bar;
    matplotlib.pyplot.show()
    print('counts in buckets')
    counts = X_cop.groupby(col).ever_60.count().reset_index().rename(columns = {'ever_60' : 'count'})
    display(pd.merge(means, counts, on = col))