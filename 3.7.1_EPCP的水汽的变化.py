# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/10/18 23:33:45 
 
@author: BUUJUN WANG
"""
#%%
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import proplot as pplt
import buujun.figure_2d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=r'.*deprecat.*')

#%%
# EPCP
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], 
                      date_parser=calc.date_parse)

def read_data(var_name, level=None):
    if level is None:
        var_array = xr.open_dataset(para.var_path[var_name])[var_name]
        if var_name=='divmf':
            var_array *= 1000
    else:
        var_array = xr.open_dataset(para.var_path[var_name])[var_name].sel(level=level)
    var_array['time'] = cnn_res.index
    return var_array

q = read_data('q').loc[:, para.lat_circ, para.lon_circ]
u850 = read_data('u', 850).loc[:, para.lat_circ, para.lon_circ]
v850 = read_data('v', 850).loc[:, para.lat_circ, para.lon_circ]

lon = q.longitude.data
lat = q.latitude.data
lonu = u850.longitude.data
latu = u850.latitude.data

#%%
# 数据切片整理

months = [4, 5]
months_condition = np.isin(cnn_res.index.month, months)

month_idx = cnn_res.index[months_condition]

epcp_idx = cnn_res.index[(months_condition) & (cnn_res.predict_ep==1)]

eper_idx = cnn_res.index[(months_condition) & (cnn_res.true_ep==1)]

# 数据切片与处理

q_yr = q.loc[month_idx].resample(time='Y').mean().loc[para.P]
u850_yr = u850.loc[month_idx].resample(time='Y').mean().loc[para.P]
v850_yr = v850.loc[month_idx].resample(time='Y').mean().loc[para.P]

q_epcp = q.loc[epcp_idx].resample(time='Y').mean().loc[para.P]
u850_epcp = u850.loc[epcp_idx].resample(time='Y').mean().loc[para.P]
v850_epcp = v850.loc[epcp_idx].resample(time='Y').mean().loc[para.P]

q_eper = q.loc[eper_idx].resample(time='Y').mean().loc[para.P]
u850_eper = u850.loc[eper_idx].resample(time='Y').mean().loc[para.P]
v850_eper = v850.loc[eper_idx].resample(time='Y').mean().loc[para.P]


#%%
kwards_geo = dict(lonlat=[100, 135, 10, 35], 
                  lonticks=np.array([100, 110, 120, 130]), 
                  latticks=np.array([10, 20, 30]))

def format_plot(ax):
    figu.geo_format(ax, **kwards_geo)
    ax.plot([para.lon_prec.start, para.lon_prec.stop, 
             para.lon_prec.stop, para.lon_prec.start, 
             para.lon_prec.start], 
            [para.lat_prec.start, para.lat_prec.start, 
             para.lat_prec.stop, para.lat_prec.stop, 
             para.lat_prec.start], color='r')

#%%
import buujun.figure_1d as figu1
importlib.reload(figu1)

q_yr_sc = q_yr.sel(longitude=slice(108, 120), latitude=slice(26.5, 21))\
    .mean(dim=['latitude','longitude'])
u850_yr_sc = u850_yr.sel(longitude=slice(108, 120), latitude=[21])\
    .mean(dim=['latitude','longitude'])
v850_yr_sc = v850_yr.sel(longitude=slice(108, 120), latitude=[21])\
    .mean(dim=['latitude','longitude'])

q_epcp_sc = q_epcp.sel(longitude=slice(108, 120), latitude=slice(26.5, 21))\
    .mean(dim=['latitude','longitude'])
u850_epcp_sc = u850_epcp.sel(longitude=slice(108, 120), latitude=[21])\
    .mean(dim=['latitude','longitude'])
v850_epcp_sc = v850_epcp.sel(longitude=slice(108, 120), latitude=[21])\
    .mean(dim=['latitude','longitude'])

year = q_yr_sc.time.dt.year

fig, axes = pplt.subplots([[1]], figsize=(11, 5))
axes.plot(year.data, q_yr_sc.data, c='grey6')
figu1.rolling_mean_plot(axes, year, q_yr_sc, demean=False)
figu1.trend_plot(axes, year, q_yr_sc, demean=False)

fig, axes = pplt.subplots([[1]], figsize=(11, 5))
axes.plot(year.data, v850_yr_sc.data, c='grey6')
figu1.rolling_mean_plot(axes, year, v850_yr_sc, demean=False)
figu1.trend_plot(axes, year, v850_yr_sc, demean=False)

fig, axes = pplt.subplots([[1]], figsize=(11, 5))
axes.plot(year.data, q_epcp_sc.data, c='grey6')
figu1.rolling_mean_plot(axes, year, q_epcp_sc, demean=False)
figu1.trend_plot(axes, year, q_epcp_sc, demean=False)

fig, axes = pplt.subplots([[1]], figsize=(11, 5))
axes.plot(year.data, v850_epcp_sc.data, c='grey6')
figu1.rolling_mean_plot(axes, year, v850_epcp_sc, demean=False)
figu1.trend_plot(axes, year, v850_epcp_sc, demean=False)

    
#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 2]], figsize=(11, 5), proj='cyl')

format_plot(axes[0])
cf = figu.trend_plot(axes[0], lon, lat, 
q_yr.rolling(time=para.rolling_window, center=True).mean(), 
                     vmin=-1, vmax=1, extend='both', 
                     cmap='precip_diff',)

figu.trend_wind_plot(axes[0], lonu, latu, 
                     u850_yr.rolling(time=para.rolling_window, center=True).mean(), 
                     v850_yr.rolling(time=para.rolling_window, center=True).mean(), 
                     scale=0.008, width=25)

format_plot(axes[1])
figu.trend_plot(axes[1], lon, lat, 
q_epcp.rolling(time=para.rolling_window, center=True).mean(), 
                vmin=-1, vmax=1, extend='both', 
                cmap='precip_diff',)

figu.trend_wind_plot(axes[1], lonu, latu, 
                     u850_epcp.rolling(time=para.rolling_window, center=True).mean(), 
                     v850_epcp.rolling(time=para.rolling_window, center=True).mean(), 
                     scale=0.01, width=25)

fig.colorbar(cf, loc='bottom', shrink=0.7)

axes[0].set_title('q&uv850 Climatology', loc='left')
axes[1].set_title('q&uv850 EPCPs', loc='left')


fig.savefig(f"./pics/FIG10_changes_q_uv850_{''.join([str(i) for i in months])}.png")
# %%
