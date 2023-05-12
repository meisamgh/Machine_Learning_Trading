
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from talib import WMA
idx= pd.IndexSlice
sns.set_style('whitegrid')
import pandas_ta as  pdta

def rank(df):
    return df.rank(axis=1, pct=True)

def scale(df):
    return df.div(df.abs().sum(axis=1), axis=0)

def log(df):
    return np.log1p(df)

def sign(df):
    return np.sign(df)

def power(df, exp):
    return df.pow(exp)

def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    return df.shift(t)

def ts_delta(df, period=1):
    return df.diff(period)

def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).sum()

def ts_mean(df, window=10):
    return df.rolling(window).mean()

def ts_weighted_mean(df, period=10):
    return (df.apply(lambda x: pdta.wma(x, period)))

def ts_std(df, window=10):
    return (df.rolling(window).std())

def ts_rank(df, window=10):
    return (df.rolling(window).apply(lambda x: x.rank().iloc[-1]))

def ts_product(df, window=10):
    return (df.rolling(window).apply(np.prod))

def ts_min(df, window=10):
    return df.rolling(window).min()

def ts_max(df, window=10):
    return df.rolling(window).max()

def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax).add(1)

def ts_argmin(df, window=10):
    return (df.rolling(window)
            .apply(np.argmin)
            .add(1))

def ts_corr(x, y, window=10):
    return x.rolling(window).corr(y)

def ts_cov(x, y, window=10):
    return x.rolling(window).cov(y)


    


