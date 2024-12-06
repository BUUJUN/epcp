# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/03/20 17:00:45 
 
@author: BUUJUN WANG

FIG 7 LRP
"""
#%%
import importlib
import torch
import numpy as np
import pandas as pd
import xarray as xr
import proplot as pplt
import cartopy.crs as ccrs
import buujun.parameters as para
import buujun.calculate as calc
import buujun.models as models
import buujun.figure_2d as figu
importlib.reload(para)
importlib.reload(calc)
importlib.reload(models)
importlib.reload(figu)

kwards_geo = dict(lonlat=[para.lon_cnn.start, para.lon_cnn.stop, 
                          para.lat_cnn.start, para.lat_cnn.stop], 
                  lonticks=np.array([100, 110, 120, 130]), 
                  latticks=np.array([5, 15, 25, 35]))

#%%
# CNN结果
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

keys = ['u', 'v', 'z']

levels = dict(u=850, v=850, z=500)

open_dataset = lambda key: xr.open_dataset(para.var_path[key])[key]\
    .sel(longitude=para.lon_cnn, latitude=para.lat_cnn, level=levels[key])

var = {key:open_dataset(key) for key in levels.keys()}

for key in keys: var[key]['time']=cnn_res.index

lon = var['z'].longitude
lat = var['z'].latitude

var_a = {
    key:var[key]-var[key].mean(dim=['time']) \
        for key in keys}

var_n = {
    key:var_a[key]/var[key].std(dim=['time'])\
        for key in keys}

epcp = cnn_res.index[cnn_res.predict_ep==1]
epe = cnn_res.index[cnn_res.true_ep==1]

var_a_epcp = {key:var_a[key].loc[epcp].data for key in keys}
var_n_epcp = {key:var_n[key].loc[epcp].data for key in keys}
var_a_epe = {key:var_a[key].loc[epe].data for key in keys}
var_n_epe = {key:var_n[key].loc[epe].data for key in keys}

feature_epcp = torch.tensor(np.stack([var_n_epcp[key] for key in keys], axis=1), 
                            dtype=torch.float, requires_grad=False)
feature_epe = torch.tensor(np.stack([var_n_epe[key] for key in keys], axis=1),
                           dtype=torch.float, requires_grad=False)

nepcp = cnn_res.index[cnn_res.predict_ep==0]
nepe = cnn_res.index[cnn_res.true_ep==0]

var_a_nepcp = {key:var_a[key].loc[nepcp].data for key in keys}
var_n_nepcp = {key:var_n[key].loc[nepcp].data for key in keys}
var_a_nepe = {key:var_a[key].loc[nepe].data for key in keys}
var_n_nepe = {key:var_n[key].loc[nepe].data for key in keys}

feature_nepcp = torch.tensor(np.stack([var_n_nepcp[key] for key in keys], axis=1), 
                            dtype=torch.float, requires_grad=False)
feature_nepe = torch.tensor(np.stack([var_n_nepe[key] for key in keys], axis=1),
                           dtype=torch.float, requires_grad=False)

#%%
net = torch.load(para.model_path).cpu()

lrp_epcp = models.LRP(feature_epcp.clone(), net).mean(axis=0)
lrp_epe = models.LRP(feature_epe.clone(), net).mean(axis=0)
lrp_nepcp = models.LRP(feature_nepcp.clone(), net).mean(axis=0)
lrp_nepe = models.LRP(feature_nepe.clone(), net).mean(axis=0)

titles =   ['LRP U850 EPCP', 'LRP V850 EPCP', 'LRP Z500 EPCP', 
            'LRP U850 REPE',  'LRP V850 REPE',  'LRP Z500 REPE', 
            'LRP U850 NEPCP','LRP V850 NEPCP','LRP Z500 NEPCP',
            'LRP U850 NREPE' ,'LRP V850 NREPE', 'LRP Z500 NREPE',]
data_lrp = [lrp_epcp[0], lrp_epcp[1], lrp_epcp[2],
            lrp_epe[0],   lrp_epe[1],  lrp_epe[2],
            lrp_nepcp[0], lrp_nepcp[1], lrp_nepcp[2], 
            lrp_nepe[0],  lrp_nepe[1], lrp_nepe[2]]
data_cir = [var_a[k] \
            for var_a in [var_a_epcp, var_a_epe, var_a_nepcp, var_a_nepe] \
                for k in keys]

#%%
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
             para.lat_prec.start], color='k')

importlib.reload(figu)
fig, axes = pplt.subplots([[1, 2, 3], 
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12]], figsize=(12, 14), proj='cyl')

cmap = pplt.Colormap("RdPu").with_extremes(under='white')
vmax_u = 0.009
vmax_v = 0.009
vmax_z = 0.0045

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i])
    if i % 3 != 0: ax.set_yticklabels([])
    if i < 9: ax.set_xticklabels([])

    if i % 3 == 0: vmax = vmax_u
    elif i % 3 == 1: vmax = vmax_v
    else: vmax = vmax_z

    pm = ax.pcolormesh(lon, lat, data_lrp[i], 
                transform=ccrs.PlateCarree(), 
                extend='neither', cmap=cmap, vmin=0, vmax=vmax)
    if i % 3 != 2: levels=np.linspace(-5, 5, 11)
    else: levels=np.linspace(-200, 200, 11)
    if i < 6:
        ax.contour(lon, lat, data_cir[i].mean(axis=0), 
                transform=ccrs.PlateCarree(), levels=levels, lw=1.5, colors='grey')
    if i % 3 == 2:
        cb = ax.colorbar(pm, loc='r', space=2)
        cb.set_ticks([cb.get_ticks()[0], cb.get_ticks()[-1]])
        cb.set_ticklabels(['low', 'high'])

import matplotlib.pyplot as plt
plt.savefig('./pics/FIG7_LRP.png')
plt.show()

# %%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 2, 3], [4, 5, 6]], figsize=(12, 8), proj='cyl')

titles = ['U850 EPCP', 'V850 EPCP', 'Z500 EPCP', 
          'U850 REPE',  'V850 REPE',  'Z500 REPE']
data_lrp = [var_n[k] \
            for var_n in [var_n_epcp, var_n_epe] \
                for k in keys]
data_cir = [var_a[k] \
            for var_a in [var_a_epcp, var_a_epe] \
                for k in keys]

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i])
    ax.contourf(lon, lat, data_lrp[i].mean(axis=0), 
                transform=ccrs.PlateCarree(), cmap='anomaly')
    ax.contour(lon, lat, data_cir[i].mean(axis=0), 
               transform=ccrs.PlateCarree(), colors='grey', lw=1)

# %%
