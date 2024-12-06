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
        if var_name=='divmf':
            circ_array = xr.open_dataset(para.var_path[var_name])[var_name]*1000
        else:
            circ_array = xr.open_dataset(para.var_path[var_name])[var_name]
    else:
        circ_array = xr.open_dataset(para.var_path[var_name])[var_name].sel(level=level)
    circ_array['time'] = cnn_res.index
    return circ_array

def diff_calculate(circ_array):
    return circ_array.loc[para.PL].mean(dim='time') - circ_array.loc[para.PE].mean(dim='time')

msl = read_dataarray('msl').loc[:, para.lat_circ, para.lon_circ]
z500 = read_dataarray('z', 500).loc[:, para.lat_circ, para.lon_circ]
u850 = read_dataarray('u', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
v850 = read_dataarray('v', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
u500 = read_dataarray('u', 500).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
v500 = read_dataarray('v', 500).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
q = read_dataarray('q').loc[:, para.lat_circ, para.lon_circ]
umf = read_dataarray('umf').loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
vmf = read_dataarray('vmf').loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]

circ_cli = [z500, msl, q]
wind_cli_u = [u500, u850, umf]
wind_cli_v = [v500, v850, vmf]

circ_sel = [z500.loc[select], msl.loc[select], q.loc[select]]
wind_sel_u = [u500.loc[select], u850.loc[select], umf.loc[select]]
wind_sel_v = [v500.loc[select], v850.loc[select], vmf.loc[select]]

lon = msl.longitude
lat = msl.latitude
lonu = u850.longitude
latu = u850.latitude

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
fig, axes = pplt.subplots([[1], [2], [3]], figsize=(6, 11), proj='cyl')

titles = ['Diff Z EPCP', 'Diff SLP EPCP', 'Diff q&mf EPCP']
units  = ['', '', '', '']
cmaps  = ['anomaly', 'purple_orange', 'precip_diff']
levels = [np.linspace(-200, 200, 11), 
          np.linspace(-50, 50, 11),
          np.linspace(-25, 25, 11)]
scales=[0.25, 0.25, 8]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                            circ_sel[i].loc[para.PE].data, 
                            circ_sel[i].loc[para.PL].data, 
                            cmap=cmaps[i], 
                            levels=levels[i],
                            extend='both', )
    figu.diff_wind_plot(ax, lonu.data, latu.data, 
                   wind_sel_u[i].loc[para.PE].data, 
                   wind_sel_v[i].loc[para.PE].data, 
                   wind_sel_u[i].loc[para.PL].data, 
                   wind_sel_v[i].loc[para.PL].data, 
                   scale=scales[i], width=5, 
                   )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')


## differences in Clim
## a) Z500 b) slp c) q  

fig, axes = pplt.subplots([[1], [2], [3]], figsize=(6, 11), proj='cyl')

titles = ['Diff Z Clim', 'Diff SLP Clim', 'Diff q&mf Clim']

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                        circ_cli[i].loc[para.PE].data, circ_cli[i].loc[para.PL].data, axis=0, 
                        cmap=cmaps[i], 
                        levels=levels[i],
                        extend='both', )
    figu.diff_wind_plot(ax, lonu.data, latu.data, 
                        wind_cli_u[i].loc[para.PE].data, wind_cli_v[i].loc[para.PE].data, 
                        wind_cli_u[i].loc[para.PL].data, 
                        wind_cli_v[i].loc[para.PL].data, 
                        scale=scales[i])
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')  


# %%
## differences in EPCPs AM
## a) Z500 b) slp c) q  

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [4, 5]))

importlib.reload(figu)
fig, axes = pplt.subplots([[1], [2], [3]], figsize=(6, 11), proj='cyl')

titles = ['Diff Z EPCP AM', 'Diff SLP EPCP AM', 'Diff q&mf EPCP AM']
units  = ['', '', '', '']
cmaps  = ['anomaly', 'purple_orange', 'precip_diff']
levels = [np.linspace(-200, 200, 11), 
          np.linspace(-50, 50, 11),
          np.linspace(-25, 25, 11)]
scales=[0.25, 0.25, 8]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                            month_sel(circ_sel[i]).loc[para.PE].data, 
                            month_sel(circ_sel[i]).loc[para.PL].data, 
                            cmap=cmaps[i], 
                            levels=levels[i],
                            extend='both', )
    figu.diff_wind_plot(ax, lonu.data, latu.data, 
                   month_sel(wind_sel_u[i]).loc[para.PE].data, 
                   month_sel(wind_sel_v[i]).loc[para.PE].data, 
                   month_sel(wind_sel_u[i]).loc[para.PL].data, 
                   month_sel(wind_sel_v[i]).loc[para.PL].data, 
                   scale=scales[i], width=5, 
                   )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')


## differences in Clim AM
## a) Z500 b) slp c) q  

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [4, 5]))

fig, axes = pplt.subplots([[1], [2], [3]], figsize=(6, 11), proj='cyl')

titles = ['Diff Z Clim AM', 'Diff SLP Clim AM', 'Diff q&mf Clim AM']

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                        month_sel(circ_cli[i]).loc[para.PE].data, 
                        month_sel(circ_cli[i]).loc[para.PL].data, axis=0, 
                        cmap=cmaps[i], 
                        levels=levels[i],
                        extend='both', )
    figu.diff_wind_plot(ax, lonu.data, latu.data, 
                        month_sel(wind_cli_u[i]).loc[para.PE].data, 
                        month_sel(wind_cli_v[i]).loc[para.PE].data, 
                        month_sel(wind_cli_u[i]).loc[para.PL].data, 
                        month_sel(wind_cli_v[i]).loc[para.PL].data, 
                        scale=scales[i])
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')  



## differences in EPCPs JJA
## a) Z500 b) slp c) q  

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [6, 7, 8]))

importlib.reload(figu)
fig, axes = pplt.subplots([[1], [2], [3]], figsize=(6, 11), proj='cyl')

titles = ['Diff Z EPCP JJA', 'Diff SLP EPCP JJA', 'Diff q&mf EPCP JJA']
units  = ['', '', '', '']
cmaps  = ['anomaly', 'purple_orange', 'precip_diff']
levels = [np.linspace(-200, 200, 11), 
          np.linspace(-50, 50, 11),
          np.linspace(-25, 25, 11)]
scales=[0.25, 0.25, 8]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                            month_sel(circ_sel[i]).loc[para.PE].data, 
                            month_sel(circ_sel[i]).loc[para.PL].data, 
                            cmap=cmaps[i], 
                            levels=levels[i],
                            extend='both', )
    figu.diff_wind_plot(ax, lonu.data, latu.data, 
                   month_sel(wind_sel_u[i]).loc[para.PE].data, 
                   month_sel(wind_sel_v[i]).loc[para.PE].data, 
                   month_sel(wind_sel_u[i]).loc[para.PL].data, 
                   month_sel(wind_sel_v[i]).loc[para.PL].data, 
                   scale=scales[i], width=5, 
                   )
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')

## differences in Clim JJA
## a) Z500 b) slp c) q  

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [6, 7, 8]))

fig, axes = pplt.subplots([[1], [2], [3]], figsize=(6, 11), proj='cyl')

titles = ['Diff Z Clim JJA', 'Diff SLP Clim JJA', 'Diff q&mf Clim JJA']

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.diff_plot(ax, lon.data, lat.data, 
                        month_sel(circ_cli[i]).loc[para.PE].data, 
                        month_sel(circ_cli[i]).loc[para.PL].data, axis=0, 
                        cmap=cmaps[i], 
                        levels=levels[i],
                        extend='both', )
    figu.diff_wind_plot(ax, lonu.data, latu.data, 
                        month_sel(wind_cli_u[i]).loc[para.PE].data, 
                        month_sel(wind_cli_v[i]).loc[para.PE].data, 
                        month_sel(wind_cli_u[i]).loc[para.PL].data, 
                        month_sel(wind_cli_v[i]).loc[para.PL].data, 
                        scale=scales[i])
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')  

#%%