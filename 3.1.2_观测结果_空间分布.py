# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/03/07 15:23:26 
 
@author: BUUJUN WANG
"""
#%%
import numpy as np
import xarray as xr
import proplot as pplt

import importlib
import buujun.figure_2d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)

# %%
ds = xr.open_dataset(para.prec_path)
lon = ds.lon
lat = ds.lat
kwards_geo = dict(lonlat=[para.lon_prec.start, para.lon_prec.stop, 
                          para.lat_prec.start, para.lat_prec.stop], 
                  lonticks=np.arange(para.lon_prec.start, para.lon_prec.stop, 4), 
                  latticks=np.arange(para.lat_prec.start, para.lat_prec.stop, 2))

#%%
prec_3d = ds.prec
expr_3d = xr.where((ds.ep_day==1)|(np.isnan(ds.ep_day)), prec_3d, 0)
exfr_3d = ds.ep_day

# %%
prec_3d_yr = para.fill_mask(prec_3d.resample(time='Y').sum())
expr_3d_yr = para.fill_mask(expr_3d.resample(time='Y').sum())
exfr_3d_yr = para.fill_mask(exfr_3d.resample(time='Y').sum())
exit_3d_yr = expr_3d_yr / exfr_3d_yr

#%%
prec_3d_yr_mean = prec_3d_yr.mean(dim='time')
expr_3d_yr_mean = expr_3d_yr.mean(dim='time')
exfr_3d_yr_mean = exfr_3d_yr.mean(dim='time')
exit_3d_yr_mean = exit_3d_yr.mean(dim='time')

prec_3d_yr_pe = prec_3d_yr.sel(time=para.PE)
expr_3d_yr_pe = expr_3d_yr.sel(time=para.PE)
exfr_3d_yr_pe = exfr_3d_yr.sel(time=para.PE)
exit_3d_yr_pe = exit_3d_yr.sel(time=para.PE)

prec_3d_yr_pl = prec_3d_yr.sel(time=para.PL)
expr_3d_yr_pl = expr_3d_yr.sel(time=para.PL)
exfr_3d_yr_pl = exfr_3d_yr.sel(time=para.PL)
exit_3d_yr_pl = exit_3d_yr.sel(time=para.PL)

# %%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 2, 5, 6], 
                           [3, 4, 7, 8]], figsize=(24, 8), proj='cyl')

def format_plot(ax, title, unit):
    figu.geo_format(ax, **kwards_geo)
    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    
def clim_plot(ax, data, levels, title, unit):
    format_plot(ax, title, unit)
    cf = figu.contourf_plot(ax, lon, lat, data, cmap='wet', levels=levels, extend='both')
    ax.colorbar(cf, space=-fig.get_size_inches()[1]/4)

data_list   = [prec_3d_yr_mean, expr_3d_yr_mean, exfr_3d_yr_mean, exit_3d_yr_mean]
level_list  = [np.linspace(800, 1400, 6), np.linspace(200, 600, 6),
               [4, 6, 7, 8, 9, 11], [40, 42, 44, 46, 48, 50]]
title_list  = ['Total prec.', 'Extreme prec.', 'Extreme freq.', 'Extreme inte.']
unit_list   = ['mm/MJJAS', 'mm/MJJAS', 'days/MJJAS', 'mm/day']

for i, data in enumerate(data_list):
    clim_plot(axes[i], data, level_list[i], title_list[i], unit_list[i])

def diff_plot(ax, data_1, data_2, levels, title, unit):
    format_plot(ax, title, unit)
    cf = figu.diff_plot(ax, lon, lat, data_1, data_2, axis=2, 
                        cmap='precip_diff2', levels=levels, extend='both')
    ax.colorbar(cf, space=-fig.get_size_inches()[1]/3) 

data_1_list   = [prec_3d_yr_pe, expr_3d_yr_pe, exfr_3d_yr_pe, exit_3d_yr_pe]
data_2_list   = [prec_3d_yr_pl, expr_3d_yr_pl, exfr_3d_yr_pl, exit_3d_yr_pl]
level_list  = [np.linspace(-250, 250, 9), np.linspace(-140, 140, 9),
               np.linspace(-3, 3, 9), np.linspace(-5.4, 5.4, 9)]
title_list  = ['Diff. total prec.', 'Diff. extreme prec.', 
               'Diff. extreme freq.', 'Diff. extreme inte.']
unit_list   = ['mm/MJJAS', 'mm/MJJAS', 'days/MJJAS', 'mm/day']

for i, data_1 in enumerate(data_1_list):
    data_2 = data_2_list[i]
    diff_plot(axes[i+4], data_1, data_2, level_list[i], title_list[i], unit_list[i])

#%%