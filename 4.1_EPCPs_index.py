# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/03/20 20:21:58 
 
@author: BUUJUN WANG
"""
#%%
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import proplot as pplt
import cartopy.crs as ccrs
from scipy import stats
from scipy import signal
import buujun.figure_1d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)

#%%
hist_name = 'pr_day_EC-Earth3_historical_r1i1p1f1_gr_1961-2014_MJJAS025x025.nc'

#%%
import os
if not os.path.exists('../data/prepared/'+hist_name):
    hist_dir = '~/Extension2/wangbj/CMIP6_025X025/historical/r1i1p1f1/'
    hist_path = hist_dir+hist_name
    hist = para.fill_mask(
        xr.open_dataset(hist_path).pr.loc[slice('1961','2014'), slice(18,26), slice(108,118)]*86400)

    hist_95 = hist.quantile(q=para.quantile, interpolation=para.interpolation)
    R95d = para.fill_mask(hist>=hist_95)
    R95d_sc = (R95d.mean(dim=['lon', 'lat'])>0.25).astype('int')
    R95t = para.fill_mask(xr.where(R95d==0, np.nan, hist).resample(time='Y').sum())
    R95t = R95t.rename(dict(time='year'))
    ds_out = xr.Dataset(dict(precip=hist, R95d=R95d, R95t=R95t, R95d_sc=R95d_sc))
    ds_out.to_netcdf(path='../data/prepared/'+hist_name)

#%%
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse).loc[slice('1961', '2014')]

epcp_indx = cnn_res.probability.groupby(cnn_res.index.year).mean()


importlib.reload(figu)
ds_obs = xr.open_dataset(para.prec_path).sel(time=slice('1961', '2014'))
expr_3d_obs = xr.where((ds_obs.R95d==1)|(np.isnan(ds_obs.R95d)), ds_obs.precip, 0)
extr_prec_obs = expr_3d_obs.mean(dim=['lon', 'lat']).resample(time='Y').sum()

hist_path = '../data/prepared/'+hist_name
ds_mod = xr.open_dataset(hist_path).sel(time=slice('1961', '2014'))
expr_3d_mod = xr.where((ds_mod.R95d==1)|(np.isnan(ds_mod.R95d)), ds_mod.precip, 0)
extr_prec_mod = expr_3d_mod.mean(dim=['lon', 'lat']).resample(time='Y').sum()

year = epcp_indx.index

fig, axes = pplt.subplots([[1], [2]], figsize=(10, 8))
figu.demean_plot(axes[0], year, extr_prec_obs.values, 
                 color_pos='pink5', color_neg='gray5')

line_color = 'bright blue'

ax02 = axes[0].alty(color=line_color)
ax02.plot(year, epcp_indx.values, color=line_color)
ax02.format(ylabel='EPCP index')
corr_obs_epcp = stats.pearsonr(extr_prec_obs.values, epcp_indx.values).statistic
axes[0].format(
    title=f'corr={corr_obs_epcp:.3f}', 
    titleloc='right', ylabel='Extreme Precip.')
axes[0].set_title('Extreme Prcip. in OBS & EPCP index', loc='left')

figu.demean_plot(axes[1], year, extr_prec_obs.values, 
                 color_pos='pink5', color_neg='gray5')
ax12 = axes[1].alty(color=line_color)
ax12.plot(year, extr_prec_mod.values, color=line_color)
ax12.format(ylabel='History')
corr_obs_mod = stats.pearsonr(extr_prec_obs.values, extr_prec_mod.values).statistic
axes[1].format(
    title=f'corr={corr_obs_mod:.3f}', 
    titleloc='right', ylabel='Extreme Precip.')
axes[1].set_title('Extreme Prcip. in OBS & History', loc='left')
print(stats.pearsonr(extr_prec_obs.values, extr_prec_mod.values))


#%%
