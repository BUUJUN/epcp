# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/27 10:44:46 
 
@author: BUUJUN WANG
"""
#%%
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, YearBegin
import xarray as xr
import proplot as pplt
from scipy import stats

import importlib
import buujun.figure_1d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)

#%%
ds = xr.open_dataset(para.prec_path).sel(time=slice('1961', '2018'))
ds = ds.sel(time=np.isin(ds.time.dt.month, [4, 5])) # 筛选四五月

demean = lambda data_array:data_array-data_array.mean(dim='time')

#%% 数据处理
ep_prec = xr.where((ds.ep_day==1)|(np.isnan(ds.ep_day)), ds.prec, np.nan)  # 极端降水的降水
to_prec_yr = para.fill_mask(ds.prec.resample(time='Y').sum()).mean(dim=['lon', 'lat']) # 降水年总降水
ep_freq_yr = para.fill_mask(ds.ep_day.resample(time='Y').sum()).mean(dim=['lon', 'lat']) #极端降水频率 
ep_prec_yr = para.fill_mask(ep_prec.resample(time='Y').sum()).mean(dim=['lon', 'lat']) # 极端降水年降水
ep_inte_yr = ep_prec_yr / ep_freq_yr # 极端降水强度

# to_prec_mon = ds.prec.mean(dim=['lon', 'lat'])\
#     .to_series().groupby(MonthBegin().rollback).sum()
# ep_freq_mon = ds.ep_day.mean(dim=['lon', 'lat'])\
#     .to_series().groupby(MonthBegin().rollback).sum()
# ep_prec_mon = ep_prec.mean(dim=['lon', 'lat'])\
#     .to_series().groupby(MonthBegin().rollback).sum()
# ep_inte_mon = ep_prec_mon / ep_freq_mon

prec_avg = ds.prec.mean(dim=['lon', 'lat'])
ep_day_sc = ds.ep_day_sc
ep_prec_sc = xr.where(ds.ep_day_sc==1, prec_avg, np.nan)

to_prec_yr_sc = prec_avg.resample(time='Y').sum()
ep_freq_yr_sc = ep_day_sc.resample(time='Y').sum()
ep_prec_yr_sc = ep_prec_sc.resample(time='Y').sum()
ep_inte_yr_sc = ep_prec_yr_sc / ep_freq_yr_sc

# to_prec_mon_sc = prec_avg.to_series().groupby(MonthBegin().rollback).sum()
# ep_freq_mon_sc = ep_day_sc.to_series().groupby(MonthBegin().rollback).sum()
# ep_prec_mon_sc = ep_prec_sc.to_series().groupby(MonthBegin().rollback).sum()
# ep_inte_mon_sc = ep_prec_mon_sc / ep_freq_mon_sc

# year = to_prec_yr.time.dt.year

# prec_freq_yr_t = (ds.prec>0).astype(int)\
#     .mean(dim=['lon', 'lat']).resample(time='Y').sum()
# prec_inte_yr_t = to_prec_yr/prec_freq_yr_t

# contrib_df = pd.DataFrame(dict(
#     prec=calc.partial_contrib(to_prec_yr, prec_freq_yr_t, prec_inte_yr_t), 
#     extreme=calc.partial_contrib(ep_prec_yr, ep_freq_yr, ep_inte_yr), 
# ))

# contrib_df_sc = pd.DataFrame(dict(
#     prec=calc.partial_contrib(to_prec_yr, prec_freq_yr_t, prec_inte_yr_t), 
#     extreme=calc.partial_contrib(ep_prec_yr_sc, ep_freq_yr_sc, ep_inte_yr_sc), 
# ))


#%%
def sum_plot(axes, data_year:xr.DataArray, ylabel_1):
    year = data_year.time.dt.year.data
    axes[:2].format(xlim=(1959, 2020))
    # axes[2].format(xlim=(4, 10))

    figu.demean_plot(axes[0], year, data_year)
    figu.rolling_mean_plot(axes[0], year, data_year)
    figu.rolling_ttest_plot(axes[1], year, data_year)
    # figu.difference_plot(axes[0], data_year)
    # figu.diff_contribute(axes[2], data_mon)

    axes[0].format(ylabel=ylabel_1, xtickloc='top', xticklabels=[])
    axes[1].format(ylabel='Significance level')
    # axes[2].format(ylabel='Contributions', xticks=np.arange(4, 9),
    #                xticklabels=['Apr', 'May', 'Jun', 'Jul', 'Aug'],
    #                yticks=[0, 0.2, 0.4, 0.6], yticklabels=['0%', '20%', '40%', '60%']) 


def intensity_plot(axes, data_year:xr.DataArray, ylabel_1):
    year = data_year.time.dt.year.data
    axes[:2].format(xlim=(1959, 2020))
    # axes[2].format(xlim=(4, 10))

    figu.demean_plot(axes[0], year, data_year)
    figu.rolling_mean_plot(axes[0], year, data_year)
    figu.rolling_ttest_plot(axes[1], year, data_year)
    figu.trend_rollmean_plot(axes[0], year, data_year)
    # figu.trend_plot(axes[0], year, data_year)
    # figu.trend_contribute_rollmean(axes[2], data_mon)

    axes[0].format(ylabel=ylabel_1, xtickloc='top', xticklabels=[])
    axes[1].format(ylabel='Significance level')
    # axes[2].format(ylabel='Contributions', xticks=np.arange(5, 10),
    #                xticklabels=['May', 'Jun', 'Jul', 'Aug', 'Sep'],
    #                yticks=[0, 0.2, 0.4, 0.6], yticklabels=['0%', '20%', '40%', '60%'])

#%%
importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1, 3, 5, 7], 
     [2, 4, 6, 8], 
    #  [9, 10, 11, 12], 
    ], figsize=(16, 6), hspace=0, dpi=100)
axes.format(abcloc='ul', abc='a')
sum_plot(axes[0:2], to_prec_yr, 'Total prec. (mm/AM)')
sum_plot(axes[2:4], ep_prec_yr, 'Total extr. prec. (mm/AM)')
sum_plot(axes[4:6], ep_freq_yr, 'EPE-R frequency (days/AM)')
intensity_plot(axes[6:8], ep_inte_yr, 'EPE-R intensity (mm/day)')
# figu.bar_plot_from_df(axes[11], contrib_df, colors=['blue7', 'red7', 'grey'])


#%%
importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1, 3, 5, 7], 
     [2, 4, 6, 8], 
    #  [9, 10, 11, 12], 
    ], figsize=(16, 6), hspace=0, dpi=100)
axes.format(abcloc='ul', abc='a')
sum_plot(axes[0:2], to_prec_yr_sc, 'Total prec. (mm/AM)')
sum_plot(axes[2:4], ep_prec_yr_sc, 'Total extr. prec. (mm/AM)')
sum_plot(axes[4:6], ep_freq_yr_sc, 'EPE-R frequency (days/AM)')
intensity_plot(axes[6:8], ep_inte_yr_sc, 'EPE-R intensity (mm/day)')
# figu.bar_plot_from_df(axes[11], contrib_df_sc, colors=['blue7', 'red7', 'grey'])






#%%
diff_to_prec_yr = to_prec_yr.loc[para.PL].mean() - to_prec_yr.loc[para.PE].mean()
diff_ep_prec_yr_contr = float((ep_prec_yr.loc[para.PL].mean() - ep_prec_yr.loc[para.PE].mean()) / diff_to_prec_yr)

diff_to_prec_yr_sc = to_prec_yr_sc.loc[para.PL].mean() - to_prec_yr_sc.loc[para.PE].mean()
diff_ep_prec_yr_contr_sc = float((ep_prec_yr_sc.loc[para.PL].mean() - ep_prec_yr_sc.loc[para.PE].mean()) / diff_to_prec_yr_sc)

diff_contr_t = pd.DataFrame(
    [[diff_ep_prec_yr_contr, diff_ep_prec_yr_contr_sc], 
     [1-diff_ep_prec_yr_contr, 1-diff_ep_prec_yr_contr_sc]], index=['extr', 'nextr'], columns=['Grid', 'SC']
)

#%%
def pdf(data, bins):
    return pd.Series(stats.gaussian_kde(data)(bins))

def dataarray_dataframe(data_array):
    data = data_array.data.reshape(data_array.shape[0], -1)
    return pd.DataFrame(data=data, index=data_array.time).dropna(axis=1)

def pdf_n(data_array, bins):
    data_frame = dataarray_dataframe(data_array)
    return data_frame.apply(lambda data:pdf(data, bins))

bins = np.linspace(0, 35, 140)
prec_pe_pdf = pdf_n(ds.prec.loc[para.PE], bins).mean(axis=1)
prec_pl_pdf = pdf_n(ds.prec.loc[para.PL], bins).mean(axis=1)

#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1]], figsize=(7, 5))

ax1 = axes
line1, = ax1.plot(bins, prec_pe_pdf, color='blue7')
line2, = ax1.plot(bins, prec_pl_pdf, color='red7')

ax1.vlines([ds.prec.where(ds.prec>1).quantile(q=0.33).data, 
            ds.prec.where(ds.prec>1).quantile(q=0.67).data], 0, 1, 
            colors='k', ls='--', lw=1.75, zorder=5)

ax1.format(
    xlim=(bins.min(), bins.max()), ylim=(0, 0.12),
    ylabel='Probability Density', xlabel='precitation (mm/day)'
)

ax2 = axes.alty(color='grey')
line3, = ax2.plot(bins, (prec_pl_pdf-prec_pe_pdf)/prec_pe_pdf, color='grey')
ax2.format(
    ylim=(-0.11, 0.11), 
    yticks=np.arange(-0.10, 0.11, 0.05),
    yticklabels=['-10%', '-5%', '0', '5%', '10%'],
    ylabel='Change'
)

axes.legend(
    handles=[line1, line2, line3], 
    labels=['PE', 'PL', 'Change'], 
    ncols=1, loc='center right')

# %%
