# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/27 21:16:02 
 
@author: BUUJUN WANG
"""
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
import buujun.parameters as para
import importlib
importlib.reload(para)

date_parse = lambda t_series: pd.to_datetime(t_series)

## ***********************
## * 一维序列计算
## ***********************

def is_one_dim(y): # 判断 y 是否为一维数组
    try: 
        if y.shape.__len__() != 1: 
            raise ValueError("Input y must be a 1-dim array.")
    except: raise ValueError("Input y must be a 1-dim array.")


def preprocessing_1d(y, index=None, fillna=False, demean=False, dropna=False):
    is_one_dim(y)
    if type(y) == xr.DataArray: 
        y = y.to_series()
    elif type(y) == np.ndarray:
        y = pd.Series(y)

    if index is not None:
        if len(index) == len(y):
            y.index = index
        else: raise Warning("Input len(index) != len(y)")

    if fillna==True: y.fillna(0, inplace=True)
    if demean==True: y=y-y.mean()
    if dropna==True: y.dropna(inplace=True)
    return y


def MKtest(y):
    def cal_uf(y):
        k = np.arange(2, y.size+1)  # k=2, 3, 4, ..., n
        e_k = k * (k+1) / 4  # 均值
        var_k = k * (k-1) * (2*k+5) / 72  # 方差
        yi, yj = np.meshgrid(y, y)
        m = (yi > yj).astype('int')
        triu = np.frompyfunc(lambda i:np.triu(m[:i+1, :i+1], k=0).sum(), 1, 1)
        s = triu(np.arange(y.size))
        uf = np.zeros_like(y)
        uf[1:] = (s[1:] - e_k) / np.sqrt(var_k)
        return uf  # uf.size == y.size
    
    y = preprocessing_1d(y)
    index_dna = y.dropna().index

    uf = y.copy()
    uf[index_dna] = cal_uf(y.dropna().values)

    ub = y.copy()
    ub[index_dna] = -cal_uf(y.dropna().values[::-1])[::-1]

    return uf, ub

def rolling_ttest(y, T=para.rolling_window):
    y = preprocessing_1d(y)
    T=int(T)

    index_dna = y.dropna().index

    rolling_mean_dna = y.dropna().rolling(window=T, center=True).mean().values
    rolling_mean = y.copy()
    rolling_mean[index_dna] = rolling_mean_dna

    rolling_std_dna = y.dropna().rolling(window=T, center=True).std().values
    rolling_std = y.copy()
    rolling_std[index_dna] = rolling_std_dna

    rolling_nobs = np.full_like(y.iloc[:-T], T)

    statistic_dna, pvalue_dna =  stats.ttest_ind_from_stats(
        mean1=rolling_mean.iloc[:-T], std1=rolling_std.iloc[:-T], nobs1=rolling_nobs,
        mean2=rolling_mean.iloc[T:], std2=rolling_std.iloc[T:], nobs2=rolling_nobs,
    ) # pvalue_dan.__len__() = data_array.__len__() - T

    pvalue = y.copy()*np.nan
    pvalue.iloc[int(np.ceil(T-T/2)):int(np.ceil(-T/2))] = pvalue_dna

    return pvalue


def filtering(y, N=2, T=para.rolling_window, method='rolling'):
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

    '''
    if method not in ['rolling', 'lowpass', 'bandpass', 'bandstop', 'highpass']:
        raise ValueError(
            "Filter method must be 'rolling', 'lowpass', 'bandpass', 'bandstop' or 'highpass'.")
    y = preprocessing_1d(y)
    index_dna = y.dropna().index

    if method in ['rolling']:
        T = int(T)    
        rolling_dna = y.dropna().rolling(window=T, center=True).mean().values
        y[index_dna] = rolling_dna
        return y
    
    from scipy import signal
    Wn = 1/T
    b, a = signal.butter(N=N, Wn=Wn, btype=method)  # 滤波器
    rolling_dna = signal.filtfilt(b, a, y.dropna())
    y[index_dna] = rolling_dna
    return y


def linregress(data):
    '''
    data.shape == (N, )
    '''
    data = np.asarray(data)
    index = np.arange(data.__len__())
    index_dpna = index[~np.isnan(data)]    
    if index_dpna.__len__() <= 1:
        return stats.linregress(index, np.ones_like(data)*np.nan)
    return stats.linregress(index_dpna, data[index_dpna])


def correlation(data1, data2):
    '''
    data.shape == (N, )
    '''
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    index = np.arange(data1.__len__())
    index_dpna = index[(~np.isnan(data1))&(~np.isnan(data2))]
    if index_dpna.__len__() <= 1:
        return stats.pearsonr([0, 0], [0, 0])
    return stats.pearsonr(data1[index_dpna], data2[index_dpna])


def correlation_n(data_array1:xr.DataArray, data_array2:xr.DataArray, dim='time'):
    return xr.apply_ufunc(
        correlation, data_array1, data_array2, 
        input_core_dims=[[dim], [dim]], 
        output_core_dims=[[], []], 
        vectorize=True)


def detrend(data):
    '''
    data.shape == (N, )
    '''
    data = np.asarray(data)
    slope, intercept, _, _, _ = linregress(data)
    return data - slope * np.arange(data.__len__()) - intercept


def demean(data, axis=0):
    data = np.asarray(data)
    return data - np.nanmean(data, axis=axis)


def partial_contrib(tota_yr, freq_yr, inte_yr):
    tota_yr = preprocessing_1d(tota_yr)
    freq_yr = preprocessing_1d(freq_yr)
    inte_yr = preprocessing_1d(inte_yr)

    freq_mean = freq_yr.mean()
    freq_dist = freq_yr - freq_yr.mean()
    inte_mean = inte_yr.mean()
    inte_dist = inte_yr - inte_yr.mean()

    freq_part = inte_mean*freq_dist
    inte_part = freq_mean*inte_dist
    noli_part= freq_dist*inte_dist

    diff_freq = freq_part.loc[para.PL].mean() - freq_part.loc[para.PE].mean()
    diff_inte = inte_part.loc[para.PL].mean() - inte_part.loc[para.PE].mean()
    diff_noli = noli_part.loc[para.PL].mean() - noli_part.loc[para.PE].mean()
    diff_tota = tota_yr.loc[para.PL].mean() - tota_yr.loc[para.PE].mean()

    contr_freq = float(diff_freq/diff_tota)
    contr_inte = float(diff_inte/diff_tota)
    contr_noli = float(diff_noli/diff_tota)

    return pd.Series(dict(Freq=contr_freq, Inte=contr_inte, Noli=contr_noli))


## ***********************
## * 多维数组计算 Xarray
## ***********************


def linregress_n(data_array:xr.DataArray, dim='time'):
    return xr.apply_ufunc(
        linregress, data_array, 
        input_core_dims=[[dim]], 
        output_core_dims=[[], [], [], [], []], 
        vectorize=True)


def detrend_n(data_array:xr.DataArray, dim='time', method='linear'):
    if method not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend method must be 'linear' or 'constant'.")
    
    if method in ['linear', 'l']:
        return xr.apply_ufunc(
            detrend, data_array, 
            input_core_dims=[[dim]], 
            output_core_dims=[[dim]], 
            vectorize=True)
    
    if method in ['constant', 'c']:
        return data_array-data_array.mean(dim=dim)
    

def anomaly_n(data_part, data_clim, axis=0):
    anomaly = data_part.mean(axis=axis) - data_clim
    pvalue = stats.ttest_1samp(a=data_part, popmean=data_clim).pvalue
    return anomaly, pvalue


def diff_n(data_1, data_2, axis=0):
    # data2 - data1
    diff = data_2.mean(axis=axis) - data_1.mean(axis=axis)
    pvalue = stats.ttest_ind(a=data_1, b=data_2, axis=axis).pvalue
    return diff, pvalue

