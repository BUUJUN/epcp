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
msl = read_dataarray('msl').loc[:, para.lat_circ, para.lon_circ]
z500 = np.divide(read_dataarray('z', 500).loc[:, para.lat_circ, para.lon_circ], para.g)
u850 = read_dataarray('u', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
v850 = read_dataarray('v', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
u500 = read_dataarray('u', 500).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
v500 = read_dataarray('v', 500).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]

lon = msl.longitude
lat = msl.latitude
lonu = u850.longitude
latu = u850.latitude

#%%
epcp = cnn_res.index[(cnn_res.predict_ep==1)&(cnn_res.true_ep==0)]
extr = cnn_res.index[(cnn_res.predict_ep==1)&(cnn_res.true_ep==1)]

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

titles = ['GPH&uv500 NREPE', 'SLP&uv850 NREPE', 
          'GPH&uv500 REPE', 'SLP&uv850 REPE']
# units  = ['$m^2 / s^2$', '$Pa$', '$m^2 / s^2$', '$Pa$']
units  = ['', '', '', '']
cmaps  = ['anomaly', 'purple_orange', 'anomaly', 'purple_orange']
levels = [np.linspace(-20, 20, 9), 
          np.linspace(-160, 160, 9),
          np.linspace(-20, 20, 9),
          np.linspace(-160, 160, 9)]

data_comp = [z500_epcp, msl_epcp, z500_extr, msl_extr]
data_clim = [z500_clim, msl_clim, z500_clim, msl_clim]
wind_comp_u = [u500_epcp, u850_epcp, u500_extr, u850_extr]
wind_comp_v = [v500_epcp, v850_epcp, v500_extr, v850_extr]
wind_clim_u = [u500_clim, u850_clim, u500_clim, u850_clim]
wind_clim_v = [v500_clim, v850_clim, v500_clim, v850_clim]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.anomaly_plot2(ax, lon, lat, data_comp[i], data_clim[i], cmap=cmaps[i], levels=levels[i], extend='both')
    figu.anomaly_wind_plot(ax, lonu, latu, wind_comp_u[i], wind_comp_v[i], wind_clim_u[i], wind_clim_v[i],
                           **kwards_wind)
    if i >= 2:
        ax.colorbar(cf, loc='right')

fig.savefig('./pics/FIGS5_EPE_NEPE.png')
fig.show()

# %%
z500_epcp_mean = z500_epcp.mean(dim='time')
u850_epcp_mean = u850_epcp.mean(dim='time')
v850_epcp_mean = v850_epcp.mean(dim='time')

z500_extr_mean = z500_extr.mean(dim='time')
u850_extr_mean = u850_extr.mean(dim='time')
v850_extr_mean = v850_extr.mean(dim='time')

z500_std = z500_extr_mean.std()
z500_mean = z500_extr_mean.mean()
u850_std = u850_extr_mean.std()
u850_mean = u850_extr_mean.mean()
v850_std = v850_extr_mean.std()
v850_mean = v850_extr_mean.mean()

# 标准化
z500_extr_nor = (z500_extr_mean-z500_mean)/z500_std
u850_extr_nor = (u850_extr_mean-u850_mean)/u850_std
v850_extr_nor = (v850_extr_mean-v850_mean)/v850_std
z500_epcp_nor = (z500_epcp_mean-z500_mean)/z500_std
u850_epcp_nor = (u850_epcp_mean-u850_mean)/u850_std
v850_epcp_nor = (v850_epcp_mean-v850_mean)/v850_std

corr_p = xr.corr(z500_extr_nor, z500_epcp_nor, ).data
corr_u = xr.corr(u850_extr_nor, u850_epcp_nor, ).data
corr_v = xr.corr(v850_extr_nor, v850_epcp_nor, ).data

rstd_p = z500_epcp_nor.std().data/z500_extr_nor.std().data
rstd_u = z500_epcp_nor.std().data/u850_extr_nor.std().data
rstd_v = v850_epcp_nor.std().data/v850_extr_nor.std().data

res_corr_std = pd.DataFrame(
    data=[[corr_p, rstd_p], 
          [corr_u, rstd_u],
          [corr_v, rstd_v]],
    columns=['corr', 'rstd'], 
    index=['Z', 'U', 'V'], dtype='float'
)

#%%
with pplt.rc.context(abc=False):
    import buujun.xrfunc as xrfunc
    importlib.reload(xrfunc)
    fig, axes = pplt.subplots([[1]], figsize=(6, 6), proj='polar')

    rticks = np.arange(0, 2, 0.5)
    rlabels = np.where(rticks==1, 'REF', rticks)
    thetaminorticks = np.append(np.arange(0, 10, 0.5)/10, np.arange(9, 10, 0.1)/10)  # 相关系数
    thetaticks = np.append(np.arange(0, 10, 1)/10, [0.95, 0.99, 1.0])  # 相关系数
    thetalabels = thetaticks

    xrfunc.set_axis(
        axes, 
        thetaticks=thetaticks, thetaticklabels=thetalabels,
        rticks=rticks, rticklabels=rlabels, 
        thetaminorticks=thetaminorticks)

    xrfunc.set_gridlines(axes, ticks_corr=np.arccos([0.9, 0.95, 0.99]), ticks_std=[0.5, 1], ticks_bias=[-0.25, 0, 0.25, 0.5, 0.75])

    markers = ('$'+res_corr_std.index.values+'$').tolist()
    labels = ['Z500', 'U850', 'V850']
    colors = ['k', 'r', 'b']

    for i, m in enumerate(markers):
        axes.scatter(np.arccos(res_corr_std['corr'].iloc[i]), 
                    res_corr_std['rstd'].iloc[i], 
                    s=100, c=colors[i], marker=m, label=labels[i])
        
    axes.legend(loc='upper left', ncols=1,  
                fc='w', ec='grey', ew=1, 
                bbox_to_anchor=(0.05, 0.85),
                borderpad=0.75)
    
    axes.set_title('EPEs V.S. NEPEs')
    fig.savefig('./pics/FIGS6_EPE_NEPE_Taylor.png')

# %%
