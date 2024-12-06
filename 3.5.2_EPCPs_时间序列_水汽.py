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

def sum_plot(axes, data_year, ylabel_1, diff=True):
    year = data_year.index.year.values
    axes[:2].format(xlim=(1959, 2020))
    # axes[2].format(xlim=(4, 10))

    figu.demean_plot(axes[0], year, data_year)
    figu.rolling_mean_plot(axes[0], year, data_year)
    figu.rolling_ttest_plot(axes[1], year, data_year)
    if diff==True: figu.difference_plot(axes[0], data_year)
    # figu.diff_contribute(axes[2], data_mon)

    axes[0].format(ylabel=ylabel_1, xticks=[], xticklabels=[])
    axes[1].format(ylabel='Significance level')
    # axes[2].format(ylabel='Contributions', xticks=np.arange(5, 10),
    #                xticklabels=['May', 'Jun', 'Jul', 'Aug', 'Sep'],
    #                yticks=[0, 0.2, 0.4, 0.6], yticklabels=['0%', '20%', '40%', '60%']) 

#%% EPCP 水汽
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

q_epcp = xr.open_dataset(para.var_path['q']).q\
    .loc[cnn_res[cnn_res.predict_ep==1].index, 
         slice(para.lat_prec.stop, para.lat_prec.start), 
         para.lon_prec].mean(dim=['latitude', 'longitude'])

q_amjja = q_epcp.sel(time=np.isin(q_epcp.time.dt.month, [4,5,6,7,8])).resample(time='Y').mean()
q_am = q_epcp.sel(time=np.isin(q_epcp.time.dt.month, [4,5])).resample(time='Y').mean()
q_jja = q_epcp.sel(time=np.isin(q_epcp.time.dt.month, [6,7,8])).resample(time='Y').mean()

year = q_amjja.time.dt.year

importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1, 3, 5],  
     [2, 4, 6]], figsize=(12, 6), hspace=(0), dpi=100)
axes.format(abcloc='ul', abc='a')
sum_plot(axes[0:2], q_amjja.to_series(), 'EPCPs q AMJJA')
sum_plot(axes[2:4], q_am.to_series(), 'EPCPs q AM', diff=False)
sum_plot(axes[4:6], q_jja.to_series(), 'EPCPs q JJA')

# figu.bar_plot_from_df(axes[11], contrib_df, colors=['blue7', 'red7', 'grey'])
# axes[0].set_title('$corr_{obs} = $'+f'${stats.pearsonr(epcp_prec_yr_t, expr_yr_t)[0]:.2f}$', loc='right')
# axes[2].set_title('$corr_{obs} = $'+f'${stats.pearsonr(epcp_freq_yr_t, exfr_yr_t)[0]:.2f}$', loc='right')
# axes[4].set_title('$corr_{obs} = $'+f'${stats.pearsonr(epcp_inte_yr_t, exit_yr_t)[0]:.2f}$', loc='right')

#%%
