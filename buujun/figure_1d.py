# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/23 17:02:29 
 
@author: BUUJUN WANG
"""
#%%
import importlib
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(para)
importlib.reload(calc)

# configuring proplot
from buujun import figure_init
importlib.reload(figure_init)

#%%
# kwards
kwards_bar = dict(width=0.8, absolute_width=True)
kwards_mean_line = dict(c='k', ls='--', lw=2, zorder=2)
kwards_trend_line = dict(c='k', ls='--', lw=2, zorder=2)
kwards_test_line = dict(c='red6', ls='-', lw=1.5, zorder=3)
kwards_zero_line = dict(c='k', ls='-', lw=1)

def kw_update(kwards_old, kwards_new):
    kwards = kwards_old.copy()
    kwards.update(kwards_new)
    return kwards


# %%
def is_one_dim(y): # 判断 y 是否为一维数组
    try: 
        if y.shape.__len__() != 1: 
            raise ValueError("Input y must be a 1-dim array.")
    except: raise ValueError("Input y must be a 1-dim array.")


def preprocessing_1d(y, fillna=False, demean=False, dropna=False, index=None, period=None):
    y = y.copy()
    is_one_dim(y)
    if type(y) == xr.DataArray: 
        y = y.to_series()
    elif type(y) == np.ndarray:
        y = pd.Series(y)
    if index is not None:
        if len(index) != len(y): 
            raise ValueError("Input len(x) != len(y)")
        y.index = index
    if period is not None: y = y.loc[period]
    if fillna==True: y.fillna(0, inplace=True)
    if demean==True: y=y-y.mean()
    if dropna==True: y.dropna(inplace=True)
    return y


def to_MultiIndex_frame(y):
    y.index = [y.index.year, y.index.month]
    y.index.names = ['year', 'month']
    data_frame = y.unstack()
    return data_frame


def pos_neg_bar_plot(axes, x, y, color_pos='red7', color_neg='blue7', ylim=None, **kwards):
    kwards_update = kw_update(kwards_bar, kwards)
    y_grt = np.where(y>=0, y, np.nan)
    y_les = np.where(y<0, y, np.nan)
    if ylim is None: ylim = (-np.abs(y).max() * 1.05, np.abs(y).max() * 1.05)
    axes.format(ylim=ylim)
    axes.bar(x, y_grt, color=color_pos, **kwards_update)
    axes.bar(x, y_les, color=color_neg, **kwards_update)
    axes.plot([axes.get_xlim()[0], axes.get_xlim()[1]], [0, 0], **kwards_zero_line)    


def demean_plot(axes, x, y, fillna=False, color_pos='red7', color_neg='blue7'):
    y = preprocessing_1d(y, fillna=fillna)
    y_dm = y - y.mean()
    pos_neg_bar_plot(axes=axes, x=x, y=y_dm, color_pos=color_pos, color_neg=color_neg)


def difference_plot(axes, y, pe=para.PE, pl=para.PL, fillna=False, demean=True, alpha=para.alpha, **kwards):
    kwards_update = kw_update(kwards_mean_line, kwards)
    y = preprocessing_1d(y, fillna=fillna, demean=demean)
    pe_series = y.loc[pe]; pe_mean = pe_series.mean()
    pl_series = y.loc[pl]; pl_mean = pl_series.mean()
    pvalue = stats.ttest_ind(pl_series, pe_series).pvalue
    if pvalue<=alpha:
        kwards_update.update(ls='-')
    else:
        kwards_update.update(ls='--')
    axes.plot([int(pe.start), int(pe.stop)],
        [pe_mean, pe_mean], **kwards_update)
    axes.plot([int(pl.start), int(pl.stop)],
        [pl_mean, pl_mean], **kwards_update)    
    print('Values of difference: ', [pe_mean, pl_mean])
    print('T-test for difference: ', pvalue, '\n')
    

def rolling_mean_plot(axes, x, y, T=para.rolling_window, fillna=False, demean=True):
    y = preprocessing_1d(y, fillna=fillna, demean=demean)
    y_rolling = calc.filtering(y, T=T)
    axes.plot(x, y_rolling, c='grey7', lw=4)
    

def rolling_ttest_plot(axes, x, y, T=para.rolling_window, alpha=para.alpha, fillna=False):
    y = preprocessing_1d(y, fillna=fillna)
    p = calc.rolling_ttest(y, T=T)
    axes.format(ylim=(-0.1, 1.1), yticks=[0, 0.5, 1], yminorticks=[])
    axes.plot(x, p, c='k', lw=2.5, zorder=1)
    axes.plot(
        [axes.get_xlim()[0], axes.get_xlim()[1]], 
        [alpha, alpha], **kwards_test_line)
    axes.fill_between(x, p, alpha, where=p<=0.05, c='red6', alpha=0.75)
    

def trend_plot_from_res(axes, idx, res_linregress, alpha=para.alpha, **kwards):
    kwards_update = kw_update(kwards_trend_line, kwards)
    pvalue = res_linregress.pvalue
    if pvalue<=alpha:
        kwards_update.update(ls='-')
    else:
        kwards_update.update(ls='--')
    axes.plot(
        [idx[0], idx[-1]],
        [res_linregress.slope*idx[0]+res_linregress.intercept, 
         res_linregress.slope*idx[-1]+res_linregress.intercept], 
        **kwards_update)
    print('T-test for trend: \n', pvalue, '\n')


def trend_plot(axes, x, y, period=para.P, fillna=False, demean=True, **kwards):
    y = preprocessing_1d(y, index=x, period=period, fillna=fillna, demean=demean, dropna=True)
    res = stats.linregress(y.index, y)
    trend_plot_from_res(axes, y.index, res, **kwards)


def trend_rollmean_plot(axes, x, y, period=para.P_trend, T=para.rolling_window, fillna=False, demean=True, **kwards):
    y = preprocessing_1d(y, index=x, period=period, fillna=fillna, demean=demean)
    if T > 1: y = calc.filtering(y, T=T).dropna()
    r1 = stats.pearsonr(y[:-1], y[1:])[0]
    df = len(y) * (1-r1) / (1+r1)
    res = stats.linregress(y.index, y, df=df)
    trend_plot_from_res(axes, y.index, res, **kwards)


def mktest_plot(axes, x, y, alpha, rolling=True, T=para.rolling_window, fillna=False):
    y = preprocessing_1d(y, fillna=fillna)
    if rolling and T > 1: y=calc.filtering(y, T=T)
    uf, ub = calc.MKtest(y)
    axes.plot(x, uf, c='red6', lw=2.5)
    axes.plot(x, ub, c='blue6', lw=2.5)
    axes.plot([axes.get_xlim()[0], axes.get_xlim()[1]], 
        [para.u_test, para.u_test], **kwards_test_line)
    axes.plot([axes.get_xlim()[0], axes.get_xlim()[1]], 
        [-para.u_test, -para.u_test], **kwards_test_line)
    umax = np.nanmax(np.abs([uf, ub]))
    axes.format(ylim=(-(1+alpha)*umax, (1+alpha)*umax), yminorticks=[])


def series_diff(series, pe=para.PE, pl=para.PL):
    pe_series = series.loc[pe]; pl_series = series.loc[pl]
    diff = pl_series.mean() - pe_series.mean()
    res = stats.ttest_ind(pe_series.dropna(), pl_series.dropna())
    return pd.Series([diff, res.statistic, res.pvalue], 
        index=['difference', 'statistic', 'pvalue'])


def diff_contribute(axes, y, pe=para.PE, pl=para.PL, fillna=False):
    y = preprocessing_1d(y, fillna=fillna)

    res_mon = to_MultiIndex_frame(y).apply(lambda series:series_diff(series, pe=pe, pl=pl))
    diff_percent = res_mon.loc['difference']/res_mon.loc['difference'].sum()
    x = diff_percent.index.values

    axes.format(xlim=(x[0]-1, x[-1]+1), xticks=x, xminorticks=[])
    axes.bar(x, diff_percent.values, color='violet5', **kwards_bar)
    axes.plot([axes.get_xlim()[0], axes.get_xlim()[1]], [0, 0], **kwards_zero_line)

    print('Values of contributions: ', diff_percent.values)
    print('T-test for monthly difference: \n', res_mon.loc['pvalue'].values, '\n')


def series_trend(series):
    res = stats.linregress(series.dropna().index, series.dropna().values)
    return pd.Series(res, index=[
        'slope', 'intercept', 'rvalue', 'pvalue', 'stderr'])


def trend_contribute(axes, y, fillna=False):
    y = preprocessing_1d(y, fillna=fillna)
    
    res_mon = to_MultiIndex_frame(y).apply(series_trend)
    trend_percent = res_mon.loc['slope']/res_mon.loc['slope'].sum()
    x = trend_percent.index.values

    axes.format(xlim=(x[0]-1, x[-1]+1), xticks=x, xminorticks=[])
    axes.bar(x, trend_percent.values, color='violet5', **kwards_bar)
    axes.plot([axes.get_xlim()[0], axes.get_xlim()[1]], [0, 0], **kwards_zero_line)

    print('Values of contributions: ', trend_percent.values)
    print('Test for monthly trend: \n', res_mon.loc['pvalue'].values, '\n')


def series_trend_rolling(series, T=para.rolling_window):
    if T <= 1: return series_trend(series)
    series_array = series.to_xarray().rename(dict(year='time'))
    rolling_array = calc.filtering(series_array, T=T)
    return series_trend(pd.Series(rolling_array, index=series.index))


def trend_contribute_rollmean(axes, y, period=para.P_trend, T=para.rolling_window, fillna=False):
    y = preprocessing_1d(y, period=period, fillna=fillna)
    
    res_mon = to_MultiIndex_frame(y).apply(lambda series:series_trend_rolling(series, T))
    trend_percent = res_mon.loc['slope']/res_mon.loc['slope'].sum()
    x = trend_percent.index.values

    axes.format(xlim=(x[0]-1, x[-1]+1), xticks=x, xminorticks=[])
    axes.bar(x, trend_percent.values, color='violet5', **kwards_bar)
    axes.plot([axes.get_xlim()[0], axes.get_xlim()[1]], [0, 0], **kwards_zero_line)

    print('Values of contributions: ', trend_percent.values)
    print('Test for monthly trend: \n', res_mon.loc['pvalue'].values, '\n')


def bar_plot_from_df(axes, dataframe, colors, xaxis='columns', legend=True, **kwards):
    kwards_update = kw_update(kwards_bar, kwards)
    if type(dataframe) != pd.DataFrame:
        raise ValueError("Input dataframe must be pd.DataFrame.")
    if xaxis not in ['columns', 'index']:
        raise ValueError("Input xaxis must be 'columns' or 'index'.")
    if xaxis == 'index': 
        dataframe = dataframe.copy().transpose()
    xticklabels=dataframe.columns.to_list()
    labels = dataframe.index.to_list()
    step = len(labels) + 1
    xlim = (0, len(xticklabels)*(step))
    xticks = np.arange(xlim[0], xlim[-1], step) + step/2
    
    if len(colors) != 1 and len(colors) != len(labels):
        raise ValueError("Input colors don't match xaxis.")
    
    axes.format(
        xlim=xlim, xticks=xticks, xticklabels=xticklabels, xminorticks=[], 
        # yticks=[0, 0.4, 0.8, 1.2], yticklabels=['0%', '40%', '80%', '120%'], 
        ylabel='Contributions')

    for i, color in enumerate(colors):
        axes.bar(np.arange(1, xlim[-1], dataframe.shape[0]+1)+i, 
                 dataframe.iloc[i], color=color, label=labels[i], **kwards_update)
    axes.plot([0, xlim[1]], [0, 0], **kwards_zero_line)
    if legend: axes.legend(ncol=1)
