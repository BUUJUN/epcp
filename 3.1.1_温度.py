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
t2m_path = '~/Extension2/wangbj/ERA5/ERA5-daily/surface/t2m_daily_1961-2018.nc'
t2m = xr.open_dataset(t2m_path).t2m\
    .sel(longitude=para.lon_prec,
         latitude=slice(26.5, 21))\
            .mean(dim=['longitude', 'latitude'])

#%%
t2m_season = t2m.resample(time='QS-DEC').mean()
t2m_DJF = t2m_season.sel(time=t2m_season.time.dt.month==12)
t2m_MAM = t2m_season.sel(time=t2m_season.time.dt.month==3)
t2m_JJA = t2m_season.sel(time=t2m_season.time.dt.month==6)
t2m_SON = t2m_season.sel(time=t2m_season.time.dt.month==9)


#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1], [2], [3], [4]], figsize=(8, 16), dpi=100)
axes.set_title('K', loc='right')


def series_plot(ax, data, title):
    year = data.time.dt.year.data
    figu.demean_plot(ax, year, data)
    figu.rolling_mean_plot(ax, year, data)
    figu.trend_rollmean_plot(ax, year, data)
    ax.set_title(title, loc='left')

series_plot(axes[0], t2m_DJF, 't2m SC DJF')
series_plot(axes[1], t2m_MAM, 't2m SC MAM')
series_plot(axes[2], t2m_JJA, 't2m SC JJA')
series_plot(axes[3], t2m_SON, 't2m SC SON')

axes.format(ylim=(-2, 2))

#%%