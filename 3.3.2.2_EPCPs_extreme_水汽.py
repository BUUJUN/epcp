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

kwards_geo = dict(lonlat=[80, 140, 0, 40], 
                  lonticks=np.array([80, 100, 120, 140]), 
                  latticks=np.array([0, 10, 20, 30, 40]))

kwards_wind = dict(width=20)

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
q = read_dataarray('q').loc[:, para.lat_circ, para.lon_circ]
divmf = read_dataarray('divmf').loc[:, para.lat_circ, para.lon_circ]
u850 = read_dataarray('u', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
v850 = read_dataarray('v', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
umf = read_dataarray('umf').loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
vmf = read_dataarray('vmf').loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]

lon = q.longitude
lat = q.latitude
lonu = u850.longitude
latu = u850.latitude

#%%
epcp = cnn_res.index[(cnn_res.predict_ep==1)&(cnn_res.true_ep==0)]
extr = cnn_res.index[(cnn_res.predict_ep==1)&(cnn_res.true_ep==1)]

q_epcp = q.loc[epcp]
divmf_epcp = divmf.loc[epcp]
u850_epcp = u850.loc[epcp]
v850_epcp = v850.loc[epcp]
umf_epcp = umf.loc[epcp]
vmf_epcp = vmf.loc[epcp]

q_extr = q.loc[extr]
divmf_extr = divmf.loc[extr]
u850_extr = u850.loc[extr]
v850_extr = v850.loc[extr]
umf_extr = umf.loc[extr]
vmf_extr = vmf.loc[extr]

q_clim = q.mean(dim='time')
divmf_clim = divmf.mean(dim='time')
u850_clim = u850.mean(dim='time')
v850_clim = v850.mean(dim='time')
umf_clim = umf.mean(dim='time')
vmf_clim = vmf.mean(dim='time')

del q, divmf, u850, v850, umf, vmf

#%%
## a) divmf EPCP; c) divmf extr 
## b) q EPCP;  d) q extr

importlib.reload(figu)
fig, axes = pplt.subplots([[1, 3], [2, 4]], figsize=(12, 8), proj='cyl')

titles = ['q&uv850 NPEPE', 'divmf&uvmf NPEPE', 
          'q&uv850 REPE', 'divmf&uvmf REPE']
# units  = ['$kg/kg$', '$kg/ (m^2 \cdot s)$', 
#           '$kg/kg$', '$kg/ (m^2 \cdot s)$']
units  = ['', '', '', '']
cmaps  = ['precip_diff', 'anomaly2', 'precip_diff', 'anomaly2']
levels = [np.linspace(-50, 50, 11), 
          np.linspace(-0.16, 0.16, 9),
          np.linspace(-50, 50, 11),
          np.linspace(-0.16, 0.16, 9)]
scales = [0.8, 30, 0.8, 30]
data_comp = [q_epcp, divmf_epcp, q_extr, divmf_extr]
data_clim = [q_clim, divmf_clim, q_clim, divmf_clim]
wind_comp_u = [u850_epcp, umf_epcp, u850_extr, umf_extr]
wind_comp_v = [v850_epcp, vmf_epcp, v850_extr, vmf_extr]
wind_clim_u = [u850_clim, umf_clim, u850_clim, umf_clim]
wind_clim_v = [v850_clim, vmf_clim, v850_clim, vmf_clim]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.anomaly_plot2(ax, lon, lat, data_comp[i], data_clim[i], cmap=cmaps[i], levels=levels[i], extend='both')
    figu.anomaly_wind_plot(
        ax, lonu, latu, wind_comp_u[i], wind_comp_v[i], wind_clim_u[i], wind_clim_v[i],
        scale=scales[i], **kwards_wind)
    if i >= 2:
        ax.colorbar(cf, loc='right')

fig.savefig('./pics/FIGS6_水汽_NREPE_REPE.png')
fig.show()

#%%