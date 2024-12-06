# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/27 10:50:57 
 
@author: BUUJUN WANG
"""
#%%
import xarray as xr
import numpy as np
import os

## 计算的参数
quantile = 0.90
interpolation = 'midpoint'
rolling_window = 9
u_test = 1.96
alpha = 0.05
g = 9.80665

## periods
P_study = slice('1961', '2018')
P = slice('1967', '2018')
PE = slice('1967', '1992')
PL = slice('1993', '2018')
P1 = slice('1967', '1992')
P2 = slice('1993', '2018')
P_trend = slice('1961', '2018')
M_study = slice(4, 8)
T_study = np.load('../data/time/DATE_1961-2018_AMJJA.npy')
T_analy = np.load('../data/time/DATE_1967-2018_AMJJA.npy')

## lon-lat
lon_prec = slice(108, 120)
lat_prec = slice(21, 26.5)
lat_prec_r = slice(26.5, 21)
lon_cnn = slice(100, 130)
lat_cnn = slice(35, 5)
lat_cnn_r = slice(5, 35)
lon_circ = slice(80, 140)
lat_circ = slice(40, 0)

### paths
## precipitation 
CN05_path = '../data/CN05.1/CN05.1_Pre_1961_2018_daily_025x025.nc'
prec_path = '../data/prepared/CN05.1_precip_1961_2018_AMJJA_daily_025x025_ETCCDI_'+str(quantile*100)+'.nc'
prec_china_path = '../data/prepared/CN05.1_precip_1961_2018_AMJJA_daily_025x025_ETCCDI_'+str(quantile*100)+'_china.nc'
mask_path = '../data/prepared/CN05.1_mask_025x025_'+str(lon_prec.start)+'-'+str(lon_prec.stop)+'_'+str(lat_prec.start)+'-'+str(lat_prec.stop)+'.nc'
if not os.path.isfile(mask_path):
    xr.where(np.isnan(xr.open_dataset(CN05_path).pre.isel(time=-1).loc[lat_prec, lon_prec]), False, True).to_netcdf(mask_path)
mask = xr.open_dataarray(mask_path)
fill_mask = lambda data_array:xr.where(mask, data_array, np.nan)

## CNN 和 result
# 1863	hpo_uv850_z_0015	187	0.957	0.3410	0.7530
# 2217	hpo_uv850_z_0018	222	0.957	0.3321	0.7463
# 4286	hpo_uv850_z_0019	429	0.957	0.3423	0.7539
# 8910	hpo_uv850_z_0095	892	0.957	0.3410	0.7530
study_name = 'hpo_uv850_z_0019'
model_file = f'hpo_uv850_z_0019_0429.pth' # √
model_dir = f'../cnn/vars_compare/data/hpo_uv850_z/'
model_path = model_dir + model_file
model_result = f'./result_{model_file}.csv'

## 大气变量场
era5_dir = '~/Extension2/wangbj/ERA5/ERA5-daily/'
var_path = dict(
    msl = era5_dir+'surface/slp_daily_1961-2018_AMJJA.nc',
    z = era5_dir+'pressure/zg_daily_1961-2018_AMJJA.nc',
    u = era5_dir+'pressure/uwind_daily_1961-2018_AMJJA.nc', 
    v = era5_dir+'pressure/vwind_daily_1961-2018_AMJJA.nc',
    q = era5_dir+'pressure/q_total_daily_1961-2018_AMJJA.nc', 
    umf = era5_dir+'pressure/mf_daily_1961-2018_AMJJA.nc', 
    vmf = era5_dir+'pressure/mf_daily_1961-2018_AMJJA.nc', 
    divmf = era5_dir+'pressure/mf_daily_1961-2018_AMJJA.nc',
    w = era5_dir+'pressure/omega_daily_1961-2018_AMJJA.nc',
    tp = era5_dir+'surface/prect_daily_1961-2018_AMJJA.nc',
    t2m = era5_dir+'surface/t2m_daily_1961-2018_AMJJA.nc',
    svo = era5_dir+'pressure/dv_850_daily_1961-2018_AMJJA.nc',
    sd = era5_dir+'pressure/dv_850_daily_1961-2018_AMJJA.nc',
)

cmip6_dir = '~/Extension2/wangbj/CMIP6/'
var_CMIP6_path = dict(
    msl = era5_dir+'surface/slp_daily_1961-2018_AMJJA.nc',
    zg = era5_dir+'pressure/zg_daily_1961-2018_AMJJA.nc',
    ua = era5_dir+'pressure/uwind_daily_1961-2018_AMJJA.nc', 
    va = era5_dir+'pressure/vwind_daily_1961-2018_AMJJA.nc',
    q = era5_dir+'pressure/q_total_daily_1961-2018_AMJJA.nc', 
    umf = era5_dir+'pressure/mf_daily_1961-2018_AMJJA.nc', 
    vmf = era5_dir+'pressure/mf_daily_1961-2018_AMJJA.nc', 
    divmf = era5_dir+'pressure/mf_daily_1961-2018_AMJJA.nc',
    w = era5_dir+'pressure/omega_daily_1961-2018_AMJJA.nc',
    tp = era5_dir+'surface/prect_daily_1961-2018_AMJJA.nc',
    t2m = era5_dir+'surface/t2m_daily_1961-2018_AMJJA.nc',
    svo = era5_dir+'pressure/dv_850_daily_1961-2018_AMJJA.nc',
    sd = era5_dir+'pressure/dv_850_daily_1961-2018_AMJJA.nc',
)

# %%
