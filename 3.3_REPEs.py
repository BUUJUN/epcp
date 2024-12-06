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
topo = xr.open_dataset('~/Extension2/wangbj/ERA5/topo.era.1.0.nc').topo.loc[para.lat_circ, para.lon_circ]
topo_plot = lambda ax: ax.contourf(topo.lon, topo.lat, topo.data, 
                                   levels=[1500, 8000, 9000, ], 
                                   colors=['grey','none'], zorder=20)

# CNN结果
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

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
    # topo_plot(ax)
    # ax.plot([para.lon_prec.start, para.lon_prec.stop, 
    #          para.lon_prec.stop, para.lon_prec.start, 
    #          para.lon_prec.start], 
    #         [para.lat_prec.start, para.lat_prec.start, 
    #          para.lat_prec.stop, para.lat_prec.stop, 
    #          para.lat_prec.start], color='r')

#%%
z500 = read_dataarray('z', 500).loc[:, para.lat_circ, para.lon_circ] / 9.8
z850 = read_dataarray('z', 850).loc[:, para.lat_circ, para.lon_circ] / 9.8
u850 = read_dataarray('u', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
v850 = read_dataarray('v', 850).loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
umf = read_dataarray('umf').loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
vmf = read_dataarray('vmf').loc[:, para.lat_circ, para.lon_circ][:, ::3, ::3]
divmf = read_dataarray('divmf').loc[:, para.lat_circ, para.lon_circ]

lon = z500.longitude
lat = z500.latitude
lonu = u850.longitude
latu = u850.latitude

#%%
import datetime
extr = cnn_res.index[cnn_res.true_ep==1]
extr_pp = extr-datetime.timedelta(4)
extr_p = extr-datetime.timedelta(2)
extr_l = extr+datetime.timedelta(2)

extr = extr[np.isin(extr_pp.month, [4,5,6,7,8])&np.isin(extr_p.month, [4,5,6,7,8])&np.isin(extr_l.month, [4,5,6,7,8])]
extr_pp = extr-datetime.timedelta(4)
extr_p = extr-datetime.timedelta(2)
extr_l = extr+datetime.timedelta(2)

extr_am = extr[np.isin(extr.month, [4,5])]
extr_pp_am = extr_am-datetime.timedelta(4)
extr_p_am = extr_am-datetime.timedelta(2)
extr_l_am = extr_am+datetime.timedelta(2)
extr_jja = extr[np.isin(extr.month, [6,7,8])]
extr_pp_jja = extr_jja-datetime.timedelta(4)
extr_p_jja = extr_jja-datetime.timedelta(2)
extr_l_jja = extr_jja+datetime.timedelta(2)

z500_extr = z500.loc[extr]
z850_extr = z850.loc[extr]
u850_extr = u850.loc[extr]
v850_extr = v850.loc[extr]
umf_extr = umf.loc[extr]
vmf_extr = vmf.loc[extr]
divmf_extr = divmf.loc[extr]

z500_extr_am= z500.loc[extr_am]
z850_extr_am= z850.loc[extr_am]
u850_extr_am= u850.loc[extr_am]
v850_extr_am= v850.loc[extr_am]
umf_extr_am= umf.loc[extr_am]
vmf_extr_am= vmf.loc[extr_am]
divmf_extr_am= divmf.loc[extr_am]

z500_extr_jja= z500.loc[extr_jja]
z850_extr_jja= z850.loc[extr_jja]
u850_extr_jja= u850.loc[extr_jja]
v850_extr_jja= v850.loc[extr_jja]
umf_extr_jja= umf.loc[extr_jja]
vmf_extr_jja= vmf.loc[extr_jja]
divmf_extr_jja= divmf.loc[extr_jja]

z500_extr_pp = z500.loc[extr_pp]
z850_extr_pp = z850.loc[extr_pp]
u850_extr_pp = u850.loc[extr_pp]
v850_extr_pp = v850.loc[extr_pp]
umf_extr_pp = umf.loc[extr_pp]
vmf_extr_pp = vmf.loc[extr_pp]
divmf_extr_pp = divmf.loc[extr_pp]

z500_extr_pp_am= z500.loc[extr_pp_am]
z850_extr_pp_am= z850.loc[extr_pp_am]
u850_extr_pp_am= u850.loc[extr_pp_am]
v850_extr_pp_am= v850.loc[extr_pp_am]
umf_extr_pp_am= umf.loc[extr_pp_am]
vmf_extr_pp_am= vmf.loc[extr_pp_am]
divmf_extr_pp_am= divmf.loc[extr_pp_am]

z500_extr_pp_jja= z500.loc[extr_pp_jja]
z850_extr_pp_jja= z850.loc[extr_pp_jja]
u850_extr_pp_jja= u850.loc[extr_pp_jja]
v850_extr_pp_jja= v850.loc[extr_pp_jja]
umf_extr_pp_jja= umf.loc[extr_pp_jja]
vmf_extr_pp_jja= vmf.loc[extr_pp_jja]
divmf_extr_pp_jja= divmf.loc[extr_pp_jja]

z500_extr_p = z500.loc[extr_p]
z850_extr_p = z850.loc[extr_p]
u850_extr_p = u850.loc[extr_p]
v850_extr_p = v850.loc[extr_p]
umf_extr_p = umf.loc[extr_p]
vmf_extr_p = vmf.loc[extr_p]
divmf_extr_p = divmf.loc[extr_p]

z500_extr_p_am= z500.loc[extr_p_am]
z850_extr_p_am= z850.loc[extr_p_am]
u850_extr_p_am= u850.loc[extr_p_am]
v850_extr_p_am= v850.loc[extr_p_am]
umf_extr_p_am= umf.loc[extr_p_am]
vmf_extr_p_am= vmf.loc[extr_p_am]
divmf_extr_p_am= divmf.loc[extr_p_am]

z500_extr_p_jja= z500.loc[extr_p_jja]
z850_extr_p_jja= z850.loc[extr_p_jja]
u850_extr_p_jja= u850.loc[extr_p_jja]
v850_extr_p_jja= v850.loc[extr_p_jja]
umf_extr_p_jja= umf.loc[extr_p_jja]
vmf_extr_p_jja= vmf.loc[extr_p_jja]
divmf_extr_p_jja= divmf.loc[extr_p_jja]

z500_extr_l = z500.loc[extr_l]
z850_extr_l = z850.loc[extr_l]
u850_extr_l = u850.loc[extr_l]
v850_extr_l = v850.loc[extr_l]
umf_extr_l = umf.loc[extr_l]
vmf_extr_l = vmf.loc[extr_l]
divmf_extr_l = divmf.loc[extr_l]

z500_extr_l_am= z500.loc[extr_l_am]
z850_extr_l_am= z850.loc[extr_l_am]
u850_extr_l_am= u850.loc[extr_l_am]
v850_extr_l_am= v850.loc[extr_l_am]
umf_extr_l_am= umf.loc[extr_l_am]
vmf_extr_l_am= vmf.loc[extr_l_am]
divmf_extr_l_am= divmf.loc[extr_l_am]

z500_extr_l_jja= z500.loc[extr_l_jja]
z850_extr_l_jja= z850.loc[extr_l_jja]
u850_extr_l_jja= u850.loc[extr_l_jja]
v850_extr_l_jja= v850.loc[extr_l_jja]
umf_extr_l_jja= umf.loc[extr_l_jja]
vmf_extr_l_jja= vmf.loc[extr_l_jja]
divmf_extr_l_jja= divmf.loc[extr_l_jja]

z500_clim = z500.mean(dim='time')
z850_clim = z850.mean(dim='time')
u850_clim = u850.mean(dim='time')
v850_clim = v850.mean(dim='time')
umf_clim = umf.mean(dim='time')
vmf_clim = vmf.mean(dim='time')
divmf_clim = divmf.mean(dim='time')

z500_clim_am = z500.sel(time=np.isin(z500.time.dt.month, [4,5])).mean(dim='time')
z850_clim_am = z850.sel(time=np.isin(z500.time.dt.month, [4,5])).mean(dim='time')
u850_clim_am = u850.sel(time=np.isin(z500.time.dt.month, [4,5])).mean(dim='time')
v850_clim_am = v850.sel(time=np.isin(z500.time.dt.month, [4,5])).mean(dim='time')
umf_clim_am = umf.sel(time=np.isin(z500.time.dt.month, [4,5])).mean(dim='time')
vmf_clim_am = vmf.sel(time=np.isin(z500.time.dt.month, [4,5])).mean(dim='time')
divmf_clim_am = divmf.sel(time=np.isin(z500.time.dt.month, [4,5])).mean(dim='time')

z500_clim_jja = z500.sel(time=np.isin(z500.time.dt.month, [6,7,8])).mean(dim='time')
z850_clim_jja = z850.sel(time=np.isin(z500.time.dt.month, [6,7,8])).mean(dim='time')
u850_clim_jja = u850.sel(time=np.isin(z500.time.dt.month, [6,7,8])).mean(dim='time')
v850_clim_jja = v850.sel(time=np.isin(z500.time.dt.month, [6,7,8])).mean(dim='time')
umf_clim_jja = umf.sel(time=np.isin(z500.time.dt.month, [6,7,8])).mean(dim='time')
vmf_clim_jja = vmf.sel(time=np.isin(z500.time.dt.month, [6,7,8])).mean(dim='time')
divmf_clim_jja = divmf.sel(time=np.isin(z500.time.dt.month, [6,7,8])).mean(dim='time')

# z500_clim_am = z500_clim
# u850_clim_am = u850_clim
# v850_clim_am = v850_clim
# umf_clim_am = umf_clim
# vmf_clim_am = vmf_clim
# divmf_clim_am = divmf_clim

# z500_clim_jja = z500_clim
# u850_clim_jja = u850_clim
# v850_clim_jja = v850_clim
# umf_clim_jja = umf_clim
# vmf_clim_jja = vmf_clim
# divmf_clim_jja = divmf_clim

# del z500, u850, v850, umf, vmf, divmf
#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 4], 
                           [2, 5],
                           [3, 6],], figsize=(12, 14), proj='cyl')

titles = ['UV850&Z500', 'UV850&Z500', 'UV850&Z500', 'UV850&Z500', 
          'MF&DivMF', 'MF&DivMF', 'MF&DivMF', 'MF&DivMF']
units  = ['-4', '-2', '0', '2', '-4',  '-2', '0', '2']
cmaps  = ['anomaly']*4+['anomaly2']*4

scales = [8]*4 + [250]*4

levels = [np.linspace(-16, 16, 11)]*4 + [np.linspace(-0.12, 0.12, 10)]*4

data_comp = [z500_extr_pp, z500_extr_p, z500_extr, z500_extr_l, 
             divmf_extr_pp, divmf_extr_p, divmf_extr, divmf_extr_l]

wind_comp_u = [u850_extr_pp, u850_extr_p, u850_extr, u850_extr_l, 
               umf_extr_pp, umf_extr_p, umf_extr, umf_extr_l]

wind_comp_v = [v850_extr_pp, v850_extr_p, v850_extr, v850_extr_l, 
               vmf_extr_pp, vmf_extr_p, vmf_extr, vmf_extr_l]

data_clim = [z500_clim, z500_clim, z500_clim, z500_clim, 
             divmf_clim, divmf_clim, divmf_clim, divmf_clim]

wind_clim_u = [u850_clim, u850_clim, u850_clim, u850_clim, 
               umf_clim, umf_clim, umf_clim, umf_clim]

wind_clim_v = [v850_clim, v850_clim, v850_clim, v850_clim, 
               vmf_clim, vmf_clim, vmf_clim, vmf_clim]

for a, i in enumerate([0, 1, 2, 4, 5, 6]):
    ax=axes[a]
    format_plot(ax, title=titles[i], unit=units[i])
    cf = figu.anomaly_plot2(ax, lon, lat, data_comp[i], data_clim[i], 
                            cmap=cmaps[i], 
                            levels=levels[i], 
                            extend='both', alpha=0.1)
    figu.anomaly_wind_plot(ax, lonu, latu, wind_comp_u[i], wind_comp_v[i], 
                           wind_clim_u[i], wind_clim_v[i], scale=scales[i])
    if (i+1) % 4 == 0:
        ax.colorbar(cf, loc='bottom')

plt.show()
# %%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 4], 
                           [2, 5],
                           [3, 6],], figsize=(12, 14), proj='cyl')

titles = ['UV850&Z500', 'UV850&Z500', 'UV850&Z500', 'UV850&Z500', 
          'MF&DivMF', 'MF&DivMF', 'MF&DivMF', 'MF&DivMF']
units  = ['-4', '-2', '0', '2', '-4',  '-2', '0', '2']
cmaps  = ['anomalyw']*4+['anomaly2']*4

scales = [3, 5, 7, 7] + [200, 250, 300, 300]

levels = [np.linspace(-15, 15, 16)]*4 + [np.linspace(-0.18, 0.18, 10)]*4
clevels = [[-15, -9, -3, 3, 9, 15]]*4 + [[-0.18, -0.06, 0.06, 0.18]]*4

data_comp = [z500_extr_pp_am, z500_extr_p_am, z500_extr_am, z500_extr_l_am, 
             divmf_extr_pp_am, divmf_extr_p_am, divmf_extr_am, divmf_extr_l_am]

data_comp_85 = [z850_extr_pp_am, z850_extr_p_am, z850_extr_am, z850_extr_l_am, 
                divmf_extr_pp_am, divmf_extr_p_am, divmf_extr_am, divmf_extr_l_am]

wind_comp_u = [u850_extr_pp_am, u850_extr_p_am, u850_extr_am, u850_extr_l_am, 
               umf_extr_pp_am, umf_extr_p_am, umf_extr_am, umf_extr_l_am]

wind_comp_v = [v850_extr_pp_am, v850_extr_p_am, v850_extr_am, v850_extr_l_am, 
               vmf_extr_pp_am, vmf_extr_p_am, vmf_extr_am, vmf_extr_l_am]

data_clim = [z500_clim_am, z500_clim_am, z500_clim_am, z500_clim_am, 
             divmf_clim_am, divmf_clim_am, divmf_clim_am, divmf_clim_am]

wind_clim_u = [u850_clim_am, u850_clim_am, u850_clim_am, u850_clim_am, 
               umf_clim_am, umf_clim_am, umf_clim_am, umf_clim_am]

wind_clim_v = [v850_clim_am, v850_clim_am, v850_clim_am, v850_clim_am, 
               vmf_clim_am, vmf_clim_am, vmf_clim_am, vmf_clim_am]

for a, i in enumerate([0, 1, 2, 4, 5, 6]):
    ax=axes[a]
    format_plot(ax, title=titles[i]+' AM', unit=units[i])
    cf = figu.anomaly_plot2(ax, lon, lat, data_comp[i], data_clim[i], 
                            cmap=cmaps[i], 
                            levels=levels[i], 
                            extend='both', alpha=1)
    if a < 3:
        figu.contour_plot(ax, lon, lat,
                          data_comp[i].mean(dim='time'), 
                          c='#FD4292', lw=3, labels=True, 
                          #   labels_kw=dict(colors='w'),
                          levels=[0, 5860, 5880], 
                          zorder=10)
    figu.anomaly_wind_plot(ax, lonu, latu, wind_comp_u[i], wind_comp_v[i], 
                           wind_clim_u[i], wind_clim_v[i], scale=scales[i], alpha=1)
    if (a+1) % 3 == 0:
        ax.colorbar(cf, loc='bottom', ticks=clevels[i])

# plt.show()
plt.savefig('./pics/FIG_3-9_REPE环流_AM.png')

#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 4], 
                           [2, 5],
                           [3, 6],], figsize=(12, 14), proj='cyl')

levels = [np.linspace(-20, 20, 16)]*4 + [np.linspace(-0.18, 0.18, 10)]*4
clevels = [[-20, -12, -4,  4, 12, 20]]*4 + [[-0.18, -0.06, 0.06, 0.18]]*4

data_comp = [z500_extr_pp_jja, z500_extr_p_jja, z500_extr_jja, z500_extr_l_jja, 
             divmf_extr_pp_jja, divmf_extr_p_jja, divmf_extr_jja, divmf_extr_l_jja]

wind_comp_u = [u850_extr_pp_jja, u850_extr_p_jja, u850_extr_jja, u850_extr_l_jja, 
               umf_extr_pp_jja, umf_extr_p_jja, umf_extr_jja, umf_extr_l_jja]

wind_comp_v = [v850_extr_pp_jja, v850_extr_p_jja, v850_extr_jja, v850_extr_l_jja, 
               vmf_extr_pp_jja, vmf_extr_p_jja, vmf_extr_jja, vmf_extr_l_jja]

data_clim = [z500_clim_jja]*4+[divmf_clim_jja]*4

wind_clim_u = [u850_clim_jja]*4+[umf_clim_jja]*4

wind_clim_v = [v850_clim_jja]*4+[vmf_clim_jja]*4

for a, i in enumerate([0, 1, 2, 4, 5, 6]):
    ax=axes[a]
    format_plot(ax, title=titles[i]+' JJA', unit=units[i])
    cf = figu.anomaly_plot2(ax, lon, lat, data_comp[i], data_clim[i], 
                            cmap=cmaps[i], 
                            levels=levels[i], 
                            extend='both', alpha=1)
    if a < 3:
        figu.contour_plot(ax, lon, lat,
                          data_comp[i].mean(dim='time'), 
                          c='#FD4292', lw=3, labels=True, 
                          #   labels_kw=dict(colors='w'),
                          levels=[0, 5860, 5880], 
                          zorder=10)
    
    figu.anomaly_wind_plot(ax, lonu, latu, wind_comp_u[i], wind_comp_v[i], 
                           wind_clim_u[i], wind_clim_v[i], scale=scales[i], alpha=1)
    if (a+1) % 3 == 0:
        ax.colorbar(cf, loc='bottom', ticks=clevels[i])

# plt.show()
plt.savefig('./pics/FIG_3-9_REPE环流_JJA.png')
# %%
