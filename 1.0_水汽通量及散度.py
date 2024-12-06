 # !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2022/12/04 21:35:45 
 
@author: BUUJUN WANG
"""
#%%
import numpy as np
import xarray as xr
import metpy.calc as mpcalc

import importlib
import buujun.parameters as para
importlib.reload(para)
#%%
print("读取数据")

era5_dir = '~/Extension2/wangbj/ERA5/ERA5-daily/'
q_path = era5_dir+'pressure/q_daily_1961-2018_AMJJA.nc'
u_path = era5_dir+'pressure/uwind_daily_1961-2018_AMJJA.nc'
v_path = era5_dir+'pressure/vwind_daily_1961-2018_AMJJA.nc'

q = xr.open_dataset(q_path).q
# u = xr.open_dataset(u_path).u
# v = xr.open_dataset(v_path).v
ds_prec = xr.open_dataset(para.prec_path)
q['time'] = ds_prec.time
# u['time'] = ds_prec.time
# v['time'] = ds_prec.time

#%%
# print("计算水汽通量")

# qu = q*u/9.80665
# qv = q*v/9.80665

# print("计算散度")

lon = q.longitude
lat = q.latitude
level = q.level

# dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat, x_dim=-1, y_dim=-2)

# divergence = lambda u, v: np.array(mpcalc.divergence(u=u, v=v, dx=dx, dy=dy, x_dim=-1, y_dim=-2))
# dims = ['latitude', 'longitude']
# div_qv = xr.apply_ufunc(divergence, 
#                         qu, qv, 
#                         input_core_dims=[dims, dims], 
#                         output_core_dims=[dims], vectorize=True)

print("计算积分")

total_q = q.isel(level=0).copy(data=np.trapz(y=q, x=level*100, axis=1))
total_q.name = 'q'
total_q.attrs.update(standard_name='total_specific_humidity', 
                     long_name='total specific humidity', 
                     units='kg kg**-1')

# total_div_qv = q.isel(level=0).copy(data=np.trapz(y=div_qv, x=level*100, axis=1))
# total_div_qv.name = 'divmf'
# total_div_qv.attrs.update(standard_name='total_divergence_moisture_flux', 
#                           long_name='divergence of total moisture flux', 
#                           units='kg m**-2 s**-1')

# total_qu = q.isel(level=0).copy(data=np.trapz(y=qu, x=level*100, axis=1))
# total_qu.name = 'umf'
# total_qu.attrs.update(standard_name='total_eastward_moisture_flux', 
#                       long_name='U component of total moisture flux', 
#                       units='kg m**-1 s**-1')

# total_qv = q.isel(level=0).copy(data=np.trapz(y=qv, x=level*100, axis=1))
# total_qv.name = 'vmf'
# total_qv.attrs.update(standard_name='total_northward_moisture_flux', 
#                       long_name='V component of total moisture flux', 
#                       units='kg m**-1 s**-1')

print("输出到文件")

# mf_path = era5_dir+'pressure/mf_daily_1961-2018_AMJJA.nc'
# xr.Dataset(data_vars=dict(divmf=total_div_qv, 
#                           umf=total_qu, 
#                           vmf=total_qv)).to_netcdf(mf_path)
xr.Dataset(data_vars=dict(q=total_q)).to_netcdf(era5_dir+'pressure/q_total_daily_1961-2018_AMJJA.nc')
