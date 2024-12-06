# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2022/10/23 14:53:43 
 
@author: BUUJUN WANG
"""
#%%
import numpy as np
import xarray as xr

import importlib
import buujun.parameters as para
importlib.reload(para)

# %%
prec = xr.open_dataset(para.CN05_path).pre.sel(time=para.P_study)

prec_summer = prec[((prec.time.dt.month>=para.M_study.start) &                                   (prec.time.dt.month<=para.M_study.stop))]

prec = prec_summer

prec_threshold = xr.where(prec>1, prec, np.nan)\
    .quantile(q=para.quantile, interpolation=para.interpolation, dim='time')

ep_day = xr.where(np.isnan(prec_summer.isel(time=-1)), 
                  np.nan, prec_summer>=prec_threshold)

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
    ep_amount=ep_amount))

#%%
# !!!!!
ds_out.to_netcdf(path=para.prec_china_path)

#%%
