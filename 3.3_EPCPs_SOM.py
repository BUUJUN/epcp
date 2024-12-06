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
from minisom import MiniSom
from partition import partition
from partition import statics

# from scipy.ndimage import zoom
# from scipy.ndimage import gaussian_filter1d
topo = xr.open_dataset('~/Extension2/wangbj/ERA5/topo.era.1.0.nc').topo.loc[para.lat_circ, para.lon_circ]
topo_mask = xr.where(np.greater_equal(topo, 3500), 1, 0)

# CNN结果
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

kwards_geo = dict(lonlat=[80, 140, 0, 40], 
                  lonticks=[], 
                  latticks=[])

PREC_ds = xr.open_dataset(para.prec_path).sel(time=slice('1967', '2018'))
P_EPE_R = PREC_ds.prec.mean(dim=['lon', 'lat']).where(PREC_ds.ep_day_sc==1)

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
    # ax.plot([para.lon_prec.start, para.lon_prec.stop, 
    #          para.lon_prec.stop, para.lon_prec.start, 
    #          para.lon_prec.start], 
    #         [para.lat_prec.start, para.lat_prec.start, 
    #          para.lat_prec.stop, para.lat_prec.stop, 
    #          para.lat_prec.start], color='r')

#%%
msl = read_dataarray('msl').sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ]
z500 = read_dataarray('z', 500).sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ] / 9.8
u850 = read_dataarray('u', 850).sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ]
v850 = read_dataarray('v', 850).sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ]
u500 = read_dataarray('u', 500).sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ]
v500 = read_dataarray('v', 500).sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ]
umf = read_dataarray('umf').sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ]
vmf = read_dataarray('vmf').sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ]
divmf = read_dataarray('divmf').sel(time=slice('1967', '2018')).loc[:, para.lat_circ, para.lon_circ]

lon = msl.longitude
lat = msl.latitude

#%%
cnn_res_sel = cnn_res.loc['1967':'2018']
epcp = cnn_res_sel.index[cnn_res_sel.predict_ep==1]

msl_epcp = msl.loc[epcp]
z500_epcp = z500.loc[epcp]
u850_epcp = u850.loc[epcp]
v850_epcp = v850.loc[epcp]
u500_epcp = u500.loc[epcp]
v500_epcp = v500.loc[epcp]

umf_epcp = umf.loc[epcp]
vmf_epcp= vmf.loc[epcp]
divmf_epcp = divmf.loc[epcp]

z500_epcp_anom = z500_epcp.copy()
u850_epcp_anom = u850_epcp.copy()
v850_epcp_anom = v850_epcp.copy()
umf_epcp_anom = umf_epcp.copy()
vmf_epcp_anom = vmf_epcp.copy()
divmf_epcp_anom = divmf_epcp.copy()

for m in [4, 5, 6, 7, 8]:
    mts = (z500.time.dt.month == m)
    mts_epcp = (z500_epcp.time.dt.month == m)

    z500_mclim = z500.loc[mts].mean(dim='time')
    z500_epcp_anom.loc[mts_epcp] = z500_epcp.loc[mts_epcp] - z500_mclim
    u850_mclim = u850.loc[mts].mean(dim='time')
    u850_epcp_anom.loc[mts_epcp] = u850_epcp.loc[mts_epcp] - u850_mclim
    v850_mclim = v850.loc[mts].mean(dim='time')
    v850_epcp_anom.loc[mts_epcp] = v850_epcp.loc[mts_epcp] - v850_mclim
    divmf_mclim = divmf.loc[mts].mean(dim='time')
    divmf_epcp_anom.loc[mts_epcp] = divmf_epcp.loc[mts_epcp] - divmf_mclim
    umf_mclim = umf.loc[mts].mean(dim='time')
    umf_epcp_anom.loc[mts_epcp] = umf_epcp.loc[mts_epcp] - umf_mclim
    vmf_mclim = vmf.loc[mts].mean(dim='time')
    vmf_epcp_anom.loc[mts_epcp] = vmf_epcp.loc[mts_epcp] - vmf_mclim

uv_len = np.sqrt(u850**2+v850**2)
uv_ang = np.arctan2(u850, v850)

uv_len_epcp = np.sqrt(u850_epcp**2+v850_epcp**2)
uv_ang_epcp = np.arctan2(u850_epcp, v850_epcp)

uv_len_epcp_anom = np.sqrt(u850_epcp_anom**2+v850_epcp_anom**2)
uv_ang_epcp_anom = np.arctan2(u850_epcp_anom, v850_epcp_anom)

#%%
# # # 标准化处理
# z500_som = z500_epcp
# uvlen_som = uv_len_epcp
# uvang_som = uv_ang_epcp

# z500_som_norm = ((z500_som - z500_som.mean(dim='time')) / z500.std(dim='time'))
# uvlen_som_norm =  ((uv_len_epcp-uv_len_epcp.mean(dim='time')) / uv_len_epcp.std(dim='time'))
# uvang_som_norm =  ((uvang_som-uvang_som.mean(dim='time')) / uvang_som.std(dim='time'))

# z500_flat = z500_som_norm.data.reshape(z500_som_norm.shape[0], -1)
# uv_len_flat = uvlen_som_norm.data.reshape(uvlen_som_norm.shape[0], -1)
# uv_ang_flat = uvang_som_norm.data.reshape(uvang_som_norm.shape[0], -1)

# data_flat = np.concatenate([z500_flat, uv_len_flat, uv_ang_flat], axis=1)

# # for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
# lr = 0.01
clsters=np.arange(9).reshape(3, 3)

# print(f'lr={lr}')

# som = MiniSom(3, 3, data_flat.shape[1], learning_rate=lr,
#             sigma=3, neighborhood_function='triangle')

# som.train(data=data_flat, num_iteration=100, 
#           random_order=False, use_epochs=True, verbose=True)

# # win_map = som.win_map(data_flat)

# def winner(data1, data2, data3, som=som, clsters=clsters):
#     data_flat = np.concatenate([data1.reshape(-1), data2.reshape(-1), data3.reshape(-1)], axis=0)
#     x, y = som.winner(data_flat)

#     return clsters[x, y]

# def winner_3d(data1, data2, data3, dim=['latitude', 'longitude']):  # 注意 dim 顺序！！
#     return xr.apply_ufunc(
#         winner, data1, data2, data3, 
#         input_core_dims=[dim, dim, dim],
#         output_core_dims=[[]],
#         vectorize=True)

# som_res = winner_3d(z500_som_norm, uvlen_som_norm, uvang_som_norm)
# som_num = som.activation_response(data_flat)

# patterns = cnn_res.predict_ep.to_xarray()
# patterns.loc[som_res.time] = som_res + 1
# patterns = patterns.sel(time=slice('1967', '2018'))
# patterns['time'] = PREC_ds.time

patterns = xr.open_dataset('pattern_dtimeepcp.nc').pattern
som_res = patterns[patterns!=0] - 1
som_num = np.array([[130, 161, 163], 
                    [128, 179, 113], 
                    [117, 134, 100]])
#%%

groupbins = som_res.time.dt.month

som_res_month_dict = dict()

for i in range(0, 9):
    som_res_month_dict[i] = (som_res==i).groupby(groupbins).sum()

som_res_month = xr.Dataset(data_vars=som_res_month_dict).to_array(dim='epoch').transpose('month', 'epoch')

pt_res = partition(patterns, P_EPE_R)

colors = ['orange', 'blue7', 'red7', 'grey']
# xr.Dataset(data_vars={'pattern': patterns}).to_netcdf('./patterns.nc')
pt_change = (pt_res.sel(time=para.P2).mean(dim='time') - pt_res.sel(time=para.P1).mean(dim='time')).to_array(dim='part')
pt_change = pt_change.transpose('pattern', 'part')
fig, axes = pplt.subplots([[1], [2]], figsize=(10, 8))#fig, axes = pplt.subplots([[1]], figsize=(12, 5))
b1 = axes[0].bar(som_res_month, cycle='rainbow', edgecolor='k', width=0.8, lw=1.25)
axes[0].format(xtickminor=False, ylim=(-2, 125), xlabel='', ylabel='', xlim=(3.5, 8.5), 
               xticks=[4,5,6,7,8], xticklabels=['April', 'May', 'June', 'July', 'August'])
# axes.bar(pt_change, cycle='rainbow', edgecolor='k', width=0.8, lw=1.25)
pt_change_t = pt_change.sel(pattern=[0]).copy()
pt_change_t.loc[0] = pt_change.sum(dim='pattern').data
pt_change_t = pt_change_t.assign_coords(pattern=[-1])
pt_change = xr.concat([pt_change_t, pt_change], dim='pattern')
b2 = axes[1].bar(pt_change[1:], cycle=colors, edgecolor='k', width=0.8, lw=1.25)
axes[1].format(xtickminor=False, xlabel='',
            xticks=range(0,10), xlim=(-0.5, 9.5), 
            xticklabels=['NEPCP']+[f'CP{i:1d}' for i in range(1,10)])
axes[0].legend(handles=b1, labels=[f'CP{i+1}' for i in range(9)], title='', ncol=3)
axes[1].legend(handles=b2, title='', ncol=2)

axes[0].set_title('Pattern distribution of each month', loc='left')
axes[1].set_title('Partition of each pattern', loc='left')

plt.savefig('./pics/FIG_补充_Partition.png')

#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1,2,3], [4,5,6], [7,8,9]], figsize=(12, 11), proj='cyl')

levels = np.linspace(-33, 33, 12)
clevels = levels

for i, ax in enumerate(axes):
    format_plot(ax, title=f'CP{i+1:1d}', 
                unit=f'{som_num[clsters==i][0]*100/som_num.sum():.0f}%')

    cf = figu.contourf_plot(ax, lon, lat, 
                            z500_epcp_anom[som_res==i].mean(dim='time'), 
                            cmap='anomalyw',
                            levels=levels, 
                            extend='both')

    figu.contour_plot(ax, lon, lat, 
                      z500_epcp[som_res==i].mean(dim='time'), 
                      c='#FD4292', 
                      lw=3,
                      labels=True, 
                    #   labels_kw=dict(colors='w'),
                      levels=[0, 5860, 5880], 
                      zorder=10)
    
    quiver = figu.wind_plot(ax, lon[::3], lat[::3], 
                           u850_epcp_anom[som_res==i].mean(dim='time')[::3, ::3], 
                           v850_epcp_anom[som_res==i].mean(dim='time')[::3, ::3], 
                           scale=15)

axes.quiverkey(quiver, 0.9, 0.05, 5, label=r'$5$ $\frac{m}{s}$', labelpos='E', 
               coordinates='figure', fontproperties=dict(size=pplt.rc['font.large']))

cb = fig.colorbar(cf, loc='bottom', shrink=0.5, label='', ticks=clevels)
axes.format(suptitle='Circulation Pattern')

plt.savefig('./pics/FIG_补充_EPCP_SOM.png')

#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1,2,3], [4,5,6], [7,8,9]], figsize=(12, 11), proj='cyl')

levels = np.linspace(-0.18, 0.18, 10)
clevels = [-0.18, -0.06, 0.06, 0.18]

for i, ax in enumerate(axes):
    format_plot(ax, title=f'CP{i+1:1d}')
    cf = figu.contourf_plot(ax, lon, lat, 
                            divmf_epcp_anom[som_res==i].mean(dim='time'), 
                            cmap='anomaly2', 
                            levels=levels, 
                            extend='both')
    quiver = figu.wind_plot(ax, lon[::3], lat[::3], 
                           umf_epcp_anom[som_res==i].mean(dim='time')[::3, ::3], 
                           vmf_epcp_anom[som_res==i].mean(dim='time')[::3, ::3], 
                           scale=600)

axes.quiverkey(quiver, 0.9, 0.05, 200, label=r'$200$ $\frac{kg}{m \cdot s}$', labelpos='E', 
               coordinates='figure', fontproperties=dict(size=pplt.rc['font.large']))
cb = fig.colorbar(cf, loc='bottom', shrink=0.5, label='', ticks=clevels)
axes.format(suptitle='Moisture Transport')

plt.savefig('./pics/FIG_补充_EPCP_MF_SOM.png')

# %%
# xr.Dataset(data_vars=dict(pattern=patterns)).to_netcdf('pattern_dtimeepcp.nc')
# %%
