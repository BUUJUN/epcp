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

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [6, 7, 8]))
select = cnn_res.index[cnn_res.predict_ep==1]

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

w500 = read_dataarray('w', 500).loc[:, para.lat_circ, para.lon_circ]
u500 = read_dataarray('u', 500).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
v500 = read_dataarray('v', 500).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
u850 = read_dataarray('u', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
v850 = read_dataarray('v', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
q = read_dataarray('q').loc[:, para.lat_circ, para.lon_circ]
umf = read_dataarray('umf').loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
vmf = read_dataarray('vmf').loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]

circ_cli = [month_sel(q), month_sel(w500), ]
wind_cli_u = [month_sel(u850), month_sel(u500), ]
wind_cli_v = [month_sel(v850), month_sel(v500), ]

circ_sel = [month_sel(q.loc[select]), month_sel(w500.loc[select]), ]
wind_sel_u = [month_sel(u850.loc[select]), month_sel(u500.loc[select]), ]
wind_sel_v = [month_sel(v850.loc[select]), month_sel(v500.loc[select]), ]

lon = w500.longitude
lat = w500.latitude
lonu = u500.longitude
latu = u500.latitude

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
fig, axes = pplt.subplots([[1, 3], [2, 4]], figsize=(11, 7), proj='cyl')

titles = ['Trend q&uv850', 'Trend w500&uv500', ]
units  = ['', '']
cmaps  = ['precip_diff', 'anomaly', ]
levels = [np.linspace(-0.01, 0.01, 11), 
          np.linspace(-0.01, 0.01, 11),]
scales=[0.2, 0.2]

# ## differences in Clim 

for i, ax in enumerate(axes[2:]):
    format_plot(ax, title=titles[i]+' Clim', unit=units[i])
    cf = figu.trend_plot(ax, lon.data, lat.data, 
                        circ_cli[i], 
                        axis=0, 
                        cmap=cmaps[i], 
                        levels=levels[i],
                        extend='both', )
    figu.diff_wind_plot(ax, lonu.data, latu.data, 
                        wind_cli_u[i].loc[para.PE].data, 
                        wind_cli_v[i].loc[para.PE].data, 
                        wind_cli_u[i].loc[para.PL].data, 
                        wind_cli_v[i].loc[para.PL].data, 
                        scale=scales[i])
    # ax.quiver(lonu.data, latu.data, wind_sel_u[i].data-wind_cli_u[i].data, wind_sel_v[i]-wind_cli_v[i])
    ax.colorbar(cf, loc='right')  

# ## differences in EPCP 

for i, ax in enumerate(axes[0:2]):
    format_plot(ax, title=titles[i]+' EPCPs', unit=units[i])
    cf = figu.trend_plot(ax, lon.data, lat.data, 
                            circ_sel[i], 
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
    # ax.colorbar(cf, loc='right')

import matplotlib.pyplot as plt
# plt.savefig('./pics/FIG10_DIFF_IN_q_w.eps', dpi=400)
# plt.savefig('./pics/FIG10_DIFF_IN_q_w.png', dpi=400)
plt.show()

#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1], [2]], figsize=(5.6, 7), proj='cyl')

format_plot(axes[0], title='Trend w500 Clim')
cf = figu.trend_plot(axes[0], lon.data, lat.data, 
                    circ_cli[1], axis=0, 
                    cmap='anomaly', levels=np.linspace(-0.03, 0.03, 13), extend='both', )
figu.diff_wind_plot(axes[0], lonu.data, latu.data, 
                wind_cli_u[1].loc[para.PE].data, 
                wind_cli_v[1].loc[para.PE].data, 
                wind_cli_u[1].loc[para.PL].data, 
                wind_cli_v[1].loc[para.PL].data, 
                scale=scales[1], width=5, )
axes[0].colorbar(cf, loc='right')

format_plot(axes[1], title='Trend q EPCPs')
cf = figu.trend_plot(axes[1], lon.data, lat.data, 
                    circ_sel[0], axis=0, 
                    cmap=cmaps[0], levels=np.linspace(-1, 1, 11), extend='both', )
figu.diff_wind_plot(axes[1], lonu.data, latu.data, 
                wind_sel_u[0].loc[para.PE].data, 
                wind_sel_v[0].loc[para.PE].data, 
                wind_sel_u[0].loc[para.PL].data, 
                wind_sel_v[0].loc[para.PL].data, 
                scale=scales[0], width=5, )
axes[1].colorbar(cf, loc='right')

# plt.savefig('./pics/FIG10_DIFF_IN_q_w_2.eps', dpi=400)
# plt.savefig('./pics/FIG10_DIFF_IN_q_w_2.png', dpi=400)
plt.show()

# %%
