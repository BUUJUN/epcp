# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2022/10/23 14:53:43 
 
@author: BUUJUN WANG
"""
#%%
import os
import numpy as np
import xarray as xr

import importlib
import buujun.parameters as para
importlib.reload(para)

# %%
prec = xr.open_dataset(para.CN05_path).pre\
    .sel(time=para.P_study, lon=para.lon_prec, lat=para.lat_prec)

prec_summer = prec[((prec.time.dt.month>=para.M_study.start) & (prec.time.dt.month<=para.M_study.stop))]

prec = prec_summer

prec_threshold = xr.where(prec>1, prec, np.nan)\
    .quantile(q=para.quantile, method=para.interpolation)


#%%
# 定义极端降水阈值  wet (> 1 mm/day) 的天的 90 分位数
ep_day = xr.where(np.isnan(prec_summer.isel(time=-1)), 
                  np.nan, prec_summer>=prec_threshold)

ep_day_avg = ep_day.sum(dim='time').where(~np.isnan(prec[0])).mean() # 平均发生EP的天数

ngrids_ep_day = ep_day.sum(dim=['lon', 'lat']) # 每天发生EP的格点数

ngrids_threshold = ngrids_ep_day.to_series()\
    .sort_values()[::-1].iloc[np.round(ep_day_avg.data).astype('int')-1]

ep_day_sc = (ngrids_ep_day>=ngrids_threshold).astype('int')

# ep_day_sc = (ep_day.mean(dim=['lon', 'lat'])>0.27225).astype('int')

print(ep_day_sc.mean().data, ep_day.mean().data)

#%%
ep_amount = xr.where(np.isnan(prec_summer.isel(time=-1)), 
                     np.nan, 
                     xr.where(ep_day==False, np.nan, prec_summer)\
                        .resample(time='Y').sum())
ep_amount = ep_amount.rename(dict(time='year'))
# ep_amount

# %%
ds_out = xr.Dataset(dict(
    prec=prec_summer, 
    prec_threshold=prec_threshold, 
    ep_day=ep_day, 
    ep_amount=ep_amount, 
    ep_day_sc=ep_day_sc))

#%% 
ds_out.to_netcdf(path=para.prec_path)
os.system(f'chmod 400 {para.prec_path}')

# %%
import xarray as xr
import buujun.parameters as para
ds_out = xr.open_dataset(para.prec_path)

# %%
