# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/27 10:44:46 
 
@author: BUUJUN WANG
"""
#%%
import numpy as np
import xarray as xr
import proplot as pplt

import importlib
import buujun.figure_1d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)

#%%
ds = xr.open_dataset(para.prec_path).sel(time=para.P_study)
demean = lambda data_array:data_array-data_array.mean(dim='time')

#%% 数据处理
# 格点发生极端降水的 amount 的区域平均
ep_prec = xr.where((ds.ep_day==1)|(np.isnan(ds.ep_day)), ds.prec, np.nan)
ep_prec_yr = para.fill_mask(ep_prec.resample(time='Y').sum())\
    .mean(dim=['lon', 'lat'])

# 区域平均降水的 极端降水部分 的 amount
prec_avg = ds.prec.mean(dim=['lon', 'lat'])
prec_avg_threshold = xr.where(prec_avg>1, prec_avg, np.nan)\
    .quantile(q=para.quantile, interpolation=para.interpolation)
ep_prec_avg_yr = xr.where(prec_avg>=prec_avg_threshold, prec_avg, np.nan).resample(time='Y').sum()

# 占比法 发生极端降水的格点的那部分极端降水平均值的 amount 
ep_day_prop = xr.where(ds.ep_day_sc==1, ds.ep_day, 0) # 华南发生极端降水时，发生极端降水的格点
ep_prec_prop = xr.where((ep_day_prop==1)|(np.isnan(ep_day_prop)), ds.prec, np.nan)
ep_prec_prop_yr = para.fill_mask(ep_prec_prop.resample(time='Y').sum())\
    .mean(dim=['lon', 'lat'])

# 占比法 发生极端降水的区域平均值的 amount 
ep_prec_prop_avg_yr = xr.where(ds.ep_day_sc==1, prec_avg, 0).resample(time='Y').sum()

#%%
def sum_plot(axes, data_year:xr.DataArray, ylabel_1):
    year = data_year.time.dt.year.data
    axes[:2].format(xlim=(1959, 2020))

    figu.demean_plot(axes[0], year, data_year)
    figu.rolling_mean_plot(axes[0], year, data_year)
    figu.rolling_ttest_plot(axes[1], year, data_year)
    # figu.difference_plot(axes[0], data_year)
    figu.trend_rollmean_plot(axes[0], year, data_year)

    axes[0].format(ylabel=ylabel_1, xtickloc='top', xticklabels=[])
    axes[1].format(ylabel='Significance level')

#%%
importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1, 3, 5], 
     [2, 4, 6], ], figsize=(15, 7), hspace=(0), dpi=100)
axes.format(abcloc='ul', abc='a')
sum_plot(axes[0:2], ep_prec_yr, 'EP amount (mm/AMJJA)')
sum_plot(axes[2:4], ep_prec_prop_avg_yr, 'EP amount prop (mm/AMJJA)')
sum_plot(axes[4:6], ep_prec_avg_yr, 'EP amount avg (mm/AMJJA)')
axes[::2].format(ylim=(-215, 215))

#%%