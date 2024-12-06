# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/27 10:44:46 
 
@author: BUUJUN WANG
"""
#%%
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import proplot as pplt
import cartopy.crs as ccrs
from scipy import stats
from scipy import signal
import buujun.figure_1d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)
from pandas.tseries.offsets import MonthBegin, YearBegin

demean = lambda data_array:data_array-data_array.mean(dim='time')

def sum_plot(axes, data_year, data_mon, ylabel_1, diff=True):
    year = data_year.index.year.values
    axes[0].format(xlim=(1959, 2020), xticks=np.arange(1960, 2021, 20))
    # axes[2].format(xlim=(4, 10))

    figu.demean_plot(axes[0], year, data_year)
    figu.rolling_mean_plot(axes[0], year, data_year)
    # figu.rolling_ttest_plot(axes[1], year, data_year)
    if diff==True: figu.difference_plot(axes[0], data_year)
    # figu.diff_contribute(axes[2], data_mon)

    axes[0].format(ylabel=ylabel_1)
    # axes[1].format(ylabel='Significance level')
    # axes[2].format(ylabel='Contributions', xticks=np.arange(5, 10),
    #                xticklabels=['May', 'Jun', 'Jul', 'Aug', 'Sep'],
    #                yticks=[0, 0.2, 0.4, 0.6], yticklabels=['0%', '20%', '40%', '60%']) 


def intensity_plot(axes, data_year:xr.DataArray, data_mon:xr.DataArray, ylabel_1):
    year = data_year.index.year.values
    axes[0].format(xlim=(1959, 2020), xticks=np.arange(1960, 2021, 20))
    # axes[2].format(xlim=(4, 10))

    figu.demean_plot(axes[0], year, data_year)
    figu.rolling_mean_plot(axes[0], year, data_year)
    # figu.rolling_ttest_plot(axes[1], year, data_year)
    # figu.difference_plot(axes[0], data_year)
    figu.trend_rollmean_plot(axes[0], year, data_year)
    # figu.trend_contribute_rollmean(axes[2], data_mon)

    axes[0].format(ylabel=ylabel_1)
    # axes[1].format(ylabel='Significance level')
    # axes[2].format(ylabel='Contributions', xticks=np.arange(5, 10),
    #                xticklabels=['May', 'Jun', 'Jul', 'Aug', 'Sep'],
    #                yticks=[0, 0.2, 0.4, 0.6], yticklabels=['0%', '20%', '40%', '60%'])

#%% 极端降水 OBS
ds = xr.open_dataset(para.prec_path).sel(time=slice('1961', '2018'))
exfr_3d = ds.ep_day
expr_3d = xr.where((ds.ep_day==1)|(np.isnan(ds.ep_day)), ds.prec, np.nan)
exfr_yr_t = para.fill_mask(exfr_3d.resample(time='Y').sum()).mean(dim=['lon', 'lat'])
expr_yr_t = para.fill_mask(expr_3d.resample(time='Y').sum()).mean(dim=['lon', 'lat'])
exit_yr_t = expr_yr_t / exfr_yr_t


#%% EPCP 降水
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

epcp_freq_1d = cnn_res.predict_ep
epcp_prec_1d = pd.Series(np.where(cnn_res.predict_ep==1, cnn_res.precipitation, np.nan), 
                         index=cnn_res.index)

epcp_freq_yr_t = epcp_freq_1d.resample('Y').sum()
epcp_prec_yr_t = epcp_prec_1d.resample('Y').sum()
epcp_inte_yr_t = epcp_prec_yr_t / epcp_freq_yr_t

epcp_freq_ymon_t = epcp_freq_1d.groupby(MonthBegin().rollback).sum()
epcp_prec_ymon_t = epcp_prec_1d.groupby(MonthBegin().rollback).sum()
epcp_inte_ymon_t = epcp_prec_ymon_t / epcp_freq_ymon_t

year = epcp_prec_yr_t.index.year


importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1, 2, 3]], figsize=(12, 4), hspace=(0))
axes.format(abcloc='ul', abc='a')
sum_plot(axes[0:1], epcp_prec_yr_t, epcp_prec_ymon_t, 'EPCPs prec. (mm/AMJJA)')
sum_plot(axes[1:2], epcp_freq_yr_t, epcp_freq_ymon_t, 'EPCPs freq. (days/AMJJA)', diff=False)
intensity_plot(axes[2:3], epcp_inte_yr_t, epcp_inte_ymon_t, 'EPCPs inte. (mm/day)')
# figu.bar_plot_from_df(axes[11], contrib_df, colors=['blue7', 'red7', 'grey'])
axes[0].set_title('$corr_{obs} = $'+f'${stats.pearsonr(epcp_prec_yr_t, expr_yr_t)[0]:.2f}$', loc='right')
axes[1].set_title('$corr_{obs} = $'+f'${stats.pearsonr(epcp_freq_yr_t, exfr_yr_t)[0]:.2f}$', loc='right')
axes[2].set_title('$corr_{obs} = $'+f'${stats.pearsonr(epcp_inte_yr_t, exit_yr_t)[0]:.2f}$', loc='right')

import matplotlib.pyplot as plt
# plt.savefig('./pics/FIG8_EPCP_PREC.png')
plt.show()

#%%
importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1, 2, 3]], figsize=(12, 4), hspace=(0))
figu.rolling_ttest_plot(axes[0], epcp_prec_yr_t.index.year, epcp_prec_yr_t)
figu.rolling_ttest_plot(axes[1], epcp_prec_yr_t.index.year, epcp_freq_yr_t)
figu.rolling_ttest_plot(axes[2], epcp_prec_yr_t.index.year, epcp_inte_yr_t)

# %%
