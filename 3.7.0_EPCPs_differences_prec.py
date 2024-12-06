# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/04/10 21:38:51 
 
@author: BUUJUN WANG
"""
#%%
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import proplot as pplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import stats
from scipy import signal
import buujun.figure_2d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)

#%%
# 数据读取与预处理
## CNN结果
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

select = cnn_res.index[(np.isin(cnn_res.index.month, [4, 5, 6, 7, 8])) & 
                       (cnn_res.predict_ep==1)]

# 环流场数据
def read_dataarray(var_name, level=None):
    if level is None:
        if var_name in ['divmf', 'tp']:
            circ_array = xr.open_dataset(para.var_path[var_name])[var_name]*1000
        else:
            circ_array = xr.open_dataset(para.var_path[var_name])[var_name]
    else:
        circ_array = xr.open_dataset(para.var_path[var_name])[var_name].sel(level=level)
    circ_array['time'] = cnn_res.index
    return circ_array

def diff_calculate(circ_array):
    return circ_array.loc[para.PL].mean(dim='time') - circ_array.loc[para.PE].mean(dim='time')

prect = read_dataarray('tp').loc[:, para.lat_circ, para.lon_circ]

circ_cli = [prect]
circ_sel = [prect.loc[select]]
lon = prect.longitude
lat = prect.latitude

# %%
# 作图
kwards_geo = dict(lonlat=[100, 135, 10, 35], 
                  lonticks=np.array([100, 110, 120, 130]), 
                  latticks=np.array([10, 20, 30]),)

def format_plot(ax, title='', unit=''):
    figu.geo_format(ax, **kwards_geo)
    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    # ax.contour(topo.lon, topo.lat, topo_mask, levels=[1], colors='g')
    ax.plot([para.lon_prec.start, para.lon_prec.stop, 
             para.lon_prec.stop, para.lon_prec.start, 
             para.lon_prec.start], 
            [para.lat_prec.start, para.lat_prec.start, 
             para.lat_prec.stop, para.lat_prec.stop, 
             para.lat_prec.start], color='r')

## differences in EPCPs
## a) Z500 b) slp c) q  

importlib.reload(figu)
fig, axes = pplt.subplots([[1]], figsize=(6, 4), proj='cyl')

titles = ['Diff P EPCP']
units  = ['']
cmaps  = ['precip_diff']
levels = [np.linspace(-0.15, 0.15, 11)]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                            circ_sel[i].loc[para.PE].data, 
                            circ_sel[i].loc[para.PL].data, 
                            cmap=cmaps[i], 
                            levels=levels[i],
                            extend='both', )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')


## differences in Clim
## a) Z500 b) slp c) q  

fig, axes = pplt.subplots([[1]], figsize=(6, 4), proj='cyl')

titles = ['Diff P Clim']

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                        circ_cli[i].loc[para.PE].data, circ_cli[i].loc[para.PL].data, axis=0, 
                        cmap=cmaps[i], 
                        levels=levels[i],
                        extend='both', )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')  


## differences in EPCPs AM
## a) Z500 b) slp c) q  

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [4, 5]))

importlib.reload(figu)
fig, axes = pplt.subplots([[1]], figsize=(6, 4), proj='cyl')

titles = ['Diff P EPCP AM']

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                            month_sel(circ_sel[i]).loc[para.PE].data, 
                            month_sel(circ_sel[i]).loc[para.PL].data, 
                            cmap=cmaps[i], 
                            levels=levels[i],
                            extend='both', )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')


## differences in Clim AM
## a) Z500 b) slp c) q  

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [4, 5]))

fig, axes = pplt.subplots([[1]], figsize=(6, 4), proj='cyl')

titles = ['Diff P Clim AM']

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                        month_sel(circ_cli[i]).loc[para.PE].data, 
                        month_sel(circ_cli[i]).loc[para.PL].data, axis=0, 
                        cmap=cmaps[i], 
                        levels=levels[i],
                        extend='both', )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')  



## differences in EPCPs JJA
## a) Z500 b) slp c) q  

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [6, 7, 8]))

importlib.reload(figu)
fig, axes = pplt.subplots([[1]], figsize=(6, 4), proj='cyl')

titles = ['Diff P EPCP JJA']

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                            month_sel(circ_sel[i]).loc[para.PE].data, 
                            month_sel(circ_sel[i]).loc[para.PL].data, 
                            cmap=cmaps[i], 
                            levels=levels[i],
                            extend='both', )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')

## differences in Clim JJA
## a) Z500 b) slp c) q  

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [6, 7, 8]))

fig, axes = pplt.subplots([[1]], figsize=(6, 4), proj='cyl')

titles = ['Diff P Clim JJA']

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                        month_sel(circ_cli[i]).loc[para.PE].data, 
                        month_sel(circ_cli[i]).loc[para.PL].data, axis=0, 
                        cmap=cmaps[i], 
                        levels=levels[i],
                        extend='both', )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')  

#%%