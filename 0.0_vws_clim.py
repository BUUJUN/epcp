# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/09/28 21:53:21 
 
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
## CNN结果
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

month_sel = lambda data_array:data_array.sel(time=np.isin(data_array.time.dt.month, [6, 7, 8]))
select = cnn_res.index[cnn_res.predict_ep==1]

vws = xr.open_dataset('../data/ERA5/surface/vws700_daily_1961-2018_AMJJA.nc').vws.loc[:,700]
vws['time'] = cnn_res.index
vws_clim = vws.mean(dim='time')
vws_diff = vws.sel(time=para.PL).mean(dim='time')-vws.sel(time=para.PE).mean(dim='time')

#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 2]], figsize=(12, 6), proj='cyl')

kwards_geo = dict(lonlat=[100, 140, 5, 35], 
                  lonticks=np.linspace(100, 140, 5), 
                  latticks=np.linspace(10, 30, 3),)

figu.geo_format(axes[0], **kwards_geo)
figu.geo_format(axes[1], **kwards_geo)

lon=vws_clim.longitude
lat=vws_clim.latitude

axes[0].contourf(lon, lat, vws_clim, levels=np.linspace(0, 10, 21), cmap='anomaly', extend='both')
axes[1].contourf(lon, lat, vws_diff, levels=np.linspace(-0.5, 0.5, 21), cmap='anomaly',  extend='both')

# %%
vws_comp = vws.loc[select].mean(dim='time')
vws_comp_diff = vws.loc[select].sel(time=para.PL).mean(dim='time')-\
    vws.loc[select].sel(time=para.PE).mean(dim='time')

importlib.reload(figu)
fig, axes = pplt.subplots([[1, 2]], figsize=(12, 6), proj='cyl')
figu.geo_format(axes[0], **kwards_geo)
figu.geo_format(axes[1], **kwards_geo)
axes[0].contourf(lon, lat, vws_comp, levels=np.linspace(0, 10, 21), cmap='anomaly', extend='both')
axes[1].contourf(lon, lat, vws_comp_diff, levels=np.linspace(-0.5, 0.5, 21), cmap='anomaly',  extend='both')

# %%
