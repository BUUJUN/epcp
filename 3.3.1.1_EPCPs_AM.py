# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/03/14 14:45:34 
 
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
# from scipy.ndimage import zoom
# from scipy.ndimage import gaussian_filter1d
# topo = xr.open_dataset('~/Extension2/wangbj/ERA5/topo.era.1.0.nc').topo.loc[para.lat_circ, para.lon_circ]
# topo_mask = xr.where(np.greater_equal(topo, 3500), 1, 0)

# CNN结果
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)
cnn_res_AM = cnn_res[np.isin(cnn_res.index.month, [4, 5])]

kwards_geo = dict(lonlat=[80, 140, 0, 40], 
                  lonticks=np.array([80, 100, 120, 140]), 
                  latticks=np.array([0, 10, 20, 30, 40]))

#%% 环流场数据
def read_dataarray(var_name, level=None):
    if level is None:
        if var_name=='divmf':
            data_array = xr.open_dataset(para.var_path[var_name])[var_name]*1000
        else:
            data_array = xr.open_dataset(para.var_path[var_name])[var_name]
    else:
        data_array = xr.open_dataset(para.var_path[var_name])[var_name].sel(level=level)
    data_array['time'] = cnn_res.index
    return data_array

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

#%%
msl = read_dataarray('msl').loc[cnn_res_AM.index, para.lat_circ, para.lon_circ]
z500 = read_dataarray('z', 500).loc[cnn_res_AM.index, para.lat_circ, para.lon_circ]
u850 = read_dataarray('u', 850).loc[cnn_res_AM.index, para.lat_circ, para.lon_circ][:, ::3, ::3]
v850 = read_dataarray('v', 850).loc[cnn_res_AM.index, para.lat_circ, para.lon_circ][:, ::3, ::3]
u500 = read_dataarray('u', 500).loc[cnn_res_AM.index, para.lat_circ, para.lon_circ][:, ::3, ::3]
v500 = read_dataarray('v', 500).loc[cnn_res_AM.index, para.lat_circ, para.lon_circ][:, ::3, ::3]

lon = msl.longitude
lat = msl.latitude
lonu = u850.longitude
latu = u850.latitude

#%%
epcp = cnn_res_AM.index[cnn_res_AM.predict_ep==1]
extr = cnn_res_AM.index[cnn_res_AM.true_ep==1]

msl_epcp = msl.loc[epcp]
z500_epcp = z500.loc[epcp]
u850_epcp = u850.loc[epcp]
v850_epcp = v850.loc[epcp]
u500_epcp = u500.loc[epcp]
v500_epcp = v500.loc[epcp]

msl_extr = msl.loc[extr]
z500_extr = z500.loc[extr]
u850_extr = u850.loc[extr]
v850_extr = v850.loc[extr]
u500_extr = u500.loc[extr]
v500_extr = v500.loc[extr]

msl_clim = msl.mean(dim='time')
z500_clim = z500.mean(dim='time')
u850_clim = u850.mean(dim='time')
v850_clim = v850.mean(dim='time')
u500_clim = u500.mean(dim='time')
v500_clim = v500.mean(dim='time')

del msl, z500, u850, v850, u500, v500

#%%
## a) Z500 EPCP; c) Z500 extr 
## b) SLP EPCP;  d) SLP extr

importlib.reload(figu)
fig, axes = pplt.subplots([[1, 3], [2, 4]], figsize=(12, 8), proj='cyl')

titles = ['Z500&uv500 EPCP AM', 'SLP&uv850 EPCP AM', 'Z500&uv500 EPE-R AM', 'SLP&uv850 EPE-R AM']
units  = ['$m^2 / s^2$', '$Pa$', '$m^2 / s^2$', '$Pa$']
cmaps  = ['anomaly', 'purple_orange', 'anomaly', 'purple_orange']
levels = [np.linspace(-200, 200, 9), 
          np.linspace(-150, 150, 7),
          np.linspace(-200, 200, 9),
          np.linspace(-150, 150, 7)]
data_comp = [z500_epcp, msl_epcp, z500_extr, msl_extr]
data_clim = [z500_clim, msl_clim, z500_clim, msl_clim]
wind_comp_u = [u500_epcp, u850_epcp, u500_extr, u850_extr]
wind_comp_v = [v500_epcp, v850_epcp, v500_extr, v850_extr]
wind_clim_u = [u500_clim, u850_clim, u500_clim, u850_clim]
wind_clim_v = [v500_clim, v850_clim, v500_clim, v850_clim]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.anomaly_plot2(ax, lon, lat, data_comp[i], data_clim[i], cmap=cmaps[i], levels=levels[i], extend='both')
    figu.anomaly_wind_plot(ax, lonu, latu, wind_comp_u[i], wind_comp_v[i], wind_clim_u[i], wind_clim_v[i])
    if i >= 2:
        ax.colorbar(cf, loc='right')

# %%
