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
# CNN结果
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

kwards_geo = dict(lonlat=[80, 140, 0, 40], 
                  lonticks=np.array([80, 100, 120, 140]), 
                  latticks=np.array([0, 10, 20, 30, 40]))

# 环流场数据
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

def diff_calculate(data_array):
    return data_array.loc[para.PL].mean(dim='time') - data_array.loc[para.PE].mean(dim='time')

def format_plot(ax, title='', unit=''):
    figu.geo_format(ax, **kwards_geo)
    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    # ax.contour(topo.lon, topo.lat, topo_mask, levels=[1], colors='g')
    ax.plot([108, 118, 118, 108, 108], [18, 18, 26, 26, 18], color='r')

#%%
msl = read_dataarray('msl').loc[:, para.LATC, para.LONC]
z500 = read_dataarray('z', 500).loc[:, para.LATC, para.LONC]
u850 = read_dataarray('u', 850).loc[:, para.LATC, para.LONC][:, ::3, ::3]
v850 = read_dataarray('v', 850).loc[:, para.LATC, para.LONC][:, ::3, ::3]
u500 = read_dataarray('u', 500).loc[:, para.LATC, para.LONC][:, ::3, ::3]
v500 = read_dataarray('v', 500).loc[:, para.LATC, para.LONC][:, ::3, ::3]
q = read_dataarray('q').loc[:, para.LATC, para.LONC]
divmf = read_dataarray('divmf').loc[:, para.LATC, para.LONC]
umf = read_dataarray('umf').loc[:, para.LATC, para.LONC][:, ::3, ::3]
vmf = read_dataarray('vmf').loc[:, para.LATC, para.LONC][:, ::3, ::3]

lon = msl.longitude
lat = msl.latitude
lonu = u850.longitude
latu = u850.latitude

epcp = cnn_res.index[cnn_res.predict_ep==1]

msl_diff_epcp = diff_calculate(msl.loc[epcp])
z500_diff_epcp = diff_calculate(z500.loc[epcp])
u850_diff_epcp = diff_calculate(u850.loc[epcp])
v850_diff_epcp = diff_calculate(v850.loc[epcp])
u500_diff_epcp = diff_calculate(u500.loc[epcp])
v500_diff_epcp = diff_calculate(v500.loc[epcp])
q_diff_epcp = diff_calculate(q.loc[epcp])
divmf_diff_epcp = diff_calculate(divmf.loc[epcp])
umf_diff_epcp = diff_calculate(umf.loc[epcp])
vmf_diff_epcp = diff_calculate(vmf.loc[epcp])

msl_diff_clim = diff_calculate(msl)
z500_diff_clim = diff_calculate(z500)
u850_diff_clim = diff_calculate(u850)
v850_diff_clim = diff_calculate(v850)
u500_diff_clim = diff_calculate(u500)
v500_diff_clim = diff_calculate(v500)
q_diff_clim = diff_calculate(q)
divmf_diff_clim = diff_calculate(divmf)
umf_diff_clim = diff_calculate(umf)
vmf_diff_clim = diff_calculate(vmf)

del msl, z500, u850, v850, u500, v500, q, divmf, umf, vmf

#%%
data_comp = [z500_diff_epcp, msl_diff_epcp, q_diff_epcp, divmf_diff_epcp]
data_clim = [z500_diff_clim, msl_diff_clim, q_diff_clim, divmf_diff_clim]
wind_comp_u = [u500_diff_epcp, u850_diff_epcp, u850_diff_epcp, umf_diff_epcp]
wind_comp_v = [v500_diff_epcp, v850_diff_epcp, v850_diff_epcp, vmf_diff_epcp]
wind_clim_u = [u500_diff_clim, u850_diff_clim, u850_diff_clim, umf_diff_clim]
wind_clim_v = [v500_diff_clim, v850_diff_clim, v850_diff_clim, vmf_diff_clim]

# %%
## a) Z500 ; c) q  
## b) slp ;  d) divmf 

importlib.reload(figu)
fig, axes = pplt.subplots([[1, 3], [2, 4]], figsize=(13, 8), proj='cyl')

titles = ['Diff Z500&uv500', 'Diff SLP&uv850', 'Diff q&uv850', 'Diff divmf&uvmf']
units  = ['', '', '', '']
cmaps  = ['anomaly', 'purple_orange', 'precip_diff', 'ColdHot']
levels = [np.linspace(-160, 160, 9), 
          np.linspace(-60, 60, 9),
          np.linspace(-40, 40, 9),
          np.linspace(-0.04, 0.04, 9)]
scales=[0.2, 0.15, 0.15, 8]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.contourf_plot(ax, lon.data, lat.data, 
                            data_comp[i].data, 
                            cmap=cmaps[i], 
                            levels=levels[i],
                            extend='both', )
    figu.wind_plot(ax, lonu.data, latu.data, 
                   wind_comp_u[i].data, 
                   wind_comp_v[i].data, 
                   scale=scales[i],
                   )
    # ax.quiver(lonu.data, latu.data, wind_comp_u[i].data-wind_clim_u[i].data, wind_comp_v[i]-wind_clim_v[i])
    ax.colorbar(cf, loc='right')
# %%
## a) Z500 ; c) q  
## b) slp ;  d) divmf 

importlib.reload(figu)
fig, axes = pplt.subplots([[1, 3], [2, 4]], figsize=(13, 8), proj='cyl')

titles = ['Diff Z500&uv500', 'Diff SLP&uv850', 'Diff q&uv850', 'Diff divmf&uvmf']
units  = ['', '', '', '']
cmaps  = ['anomaly', 'purple_orange', 'precip_diff', 'ColdHot']
levels = [np.linspace(-100, 100, 11), 
          np.linspace(-100, 100, 11),
          np.linspace(-20, 20, 11),
          np.linspace(-0.05, 0.05, 11)]
scales=[0.2, 0.15, 0.15, 6]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.contourf_plot(ax, lon.data, lat.data, 
                            data_comp[i].data-data_clim[i].data, 
                            cmap=cmaps[i], 
                            levels=levels[i],
                            extend='both', )
    figu.wind_plot(ax, lonu.data, latu.data, 
                   wind_comp_u[i].data-wind_clim_u[i].data, 
                   wind_comp_v[i].data-wind_clim_v[i].data, scale=scales[i ])
    # ax.quiver(lonu.data, latu.data, wind_comp_u[i].data-wind_clim_u[i].data, wind_comp_v[i]-wind_clim_v[i])
    ax.colorbar(cf, loc='right')  

#%%