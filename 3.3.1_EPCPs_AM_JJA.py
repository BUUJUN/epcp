# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/10/29 16:47:31 
 
@author: BUUJUN WANG
"""
#%%
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import proplot as pplt
from scipy import stats
import buujun.figure_2d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=r'.*deprecat.*')

def read_data(name, level=None):
    if level is None:
        array = xr.open_dataset(para.var_path[name])[name]
        if name=='divmf': array *= 1000
    else:
        array = xr.open_dataset(para.var_path[name])[name].sel(level=level)
    array['time'] = para.T_study
    array.transpose('time', 'latitude', 'longitude')
    return array

def geo_format(ax, title, unit):
    kwards_geo = dict(lonlat=[80, 140, 0, 40], 
                  lonticks=np.array([80, 100, 120, 140]), 
                  latticks=np.array([0, 10, 20, 30, 40]))
    figu.geo_format(ax, **kwards_geo)
    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    # ax.contour(topo.lon, topo.lat, topo_mask, levels=[1], colors='g')
    ax.plot([para.lon_prec.start, para.lon_prec.stop, 
             para.lon_prec.stop, para.lon_prec.start, 
             para.lon_prec.start], 
            [para.lat_prec.start, para.lat_prec.start, 
             para.lat_prec.stop, para.lat_prec.stop, 
             para.lat_prec.start], color='r')

#%%
# 常量定义
AM=[4,5]
JJA=[6,7,8]

CIRC_SEL=dict(time=slice('1967-01-01', '2018-12-31'), 
              latitude=para.lat_circ, longitude=para.lon_circ)

wind_kwards=dict(width=20)
cf_kwards=dict(extend='both')

#%%
# 数据读取
CNN_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)
CNN_res = CNN_res.loc['1967-01-01':'2018-12-31']

U850=read_data(name='u', level=850).sel(CIRC_SEL)[:,::3,::3]
V850=read_data(name='v', level=850).sel(CIRC_SEL)[:,::3,::3]
U500=read_data(name='u', level=500).sel(CIRC_SEL)[:,::3,::3]
V500=read_data(name='v', level=500).sel(CIRC_SEL)[:,::3,::3]
Z = np.divide(read_data(name='z', level=500).sel(CIRC_SEL), para.g)
SLP=read_data(name='msl').sel(CIRC_SEL)
UMF=read_data(name='umf').sel(CIRC_SEL)[:,::3,::3]
VMF=read_data(name='vmf').sel(CIRC_SEL)[:,::3,::3]
Q=read_data(name='q').sel(CIRC_SEL)
DIVMF=read_data(name='divmf').sel(CIRC_SEL)

lon=Z.longitude
lat=Z.latitude
lonu=U850.longitude
latu=U850.latitude

Cond_AM = np.isin(CNN_res.index.month, AM)
Cond_JJA = np.isin(CNN_res.index.month, JJA)
Cond_EPCP = CNN_res.predict_ep==1
Cond_REPE = CNN_res.true_ep==1

#%%
# 气候态
U850_clim=U850.mean(dim='time')
V850_clim=V850.mean(dim='time')
U500_clim=U500.mean(dim='time')
V500_clim=V500.mean(dim='time')
Z_clim=Z.mean(dim='time')
SLP_clim=SLP.mean(dim='time')
UMF_clim=UMF.mean(dim='time')
VMF_clim=VMF.mean(dim='time')
Q_clim=Q.mean(dim='time')

# %%
levels_slp = np.linspace(-200, 200, 11)
levels_z = np.linspace(-20, 20, 11)
levels_q = np.linspace(-60, 60, 9)

AM_EPCP_idx = CNN_res.index[Cond_AM & Cond_EPCP]
AM_REPE_idx = CNN_res.index[Cond_AM & Cond_REPE]
AM_idx = CNN_res.index[Cond_AM]

U850_AM = U850.loc[AM_idx].mean(dim='time')
V850_AM = V850.loc[AM_idx].mean(dim='time')
SLP_AM = SLP.loc[AM_idx].mean(dim='time')
U500_AM = U500.loc[AM_idx].mean(dim='time')
V500_AM = V500.loc[AM_idx].mean(dim='time')
Z_AM = Z.loc[AM_idx].mean(dim='time')
UMF_AM = UMF.loc[AM_idx].mean(dim='time')
VMF_AM = VMF.loc[AM_idx].mean(dim='time')
Q_AM = Q.loc[AM_idx].mean(dim='time')
DIVMF_AM = DIVMF.loc[AM_idx].mean(dim='time')

U850_EPCP_AM = U850.loc[AM_EPCP_idx]#.mean(dim='time')
V850_EPCP_AM = V850.loc[AM_EPCP_idx]#.mean(dim='time')
SLP_EPCP_AM = SLP.loc[AM_EPCP_idx]#.mean(dim='time')
U500_EPCP_AM = U500.loc[AM_EPCP_idx]#.mean(dim='time')
V500_EPCP_AM = V500.loc[AM_EPCP_idx]#.mean(dim='time')
Z_EPCP_AM = Z.loc[AM_EPCP_idx]#.mean(dim='time')
UMF_EPCP_AM = UMF.loc[AM_EPCP_idx]#.mean(dim='time')
VMF_EPCP_AM = VMF.loc[AM_EPCP_idx]#.mean(dim='time')
Q_EPCP_AM = Q.loc[AM_EPCP_idx]#.mean(dim='time')
DIVMF_EPCP_AM = DIVMF.loc[AM_EPCP_idx]#.mean(dim='time')

U850_REPE_AM = U850.loc[AM_REPE_idx]#.mean(dim='time')
V850_REPE_AM = V850.loc[AM_REPE_idx]#.mean(dim='time')
SLP_REPE_AM = SLP.loc[AM_REPE_idx]#.mean(dim='time')
U500_REPE_AM = U500.loc[AM_REPE_idx]#.mean(dim='time')
V500_REPE_AM = V500.loc[AM_REPE_idx]#.mean(dim='time')
Z_REPE_AM = Z.loc[AM_REPE_idx]#.mean(dim='time')
UMF_REPE_AM = UMF.loc[AM_REPE_idx]#.mean(dim='time')
VMF_REPE_AM = VMF.loc[AM_REPE_idx]#.mean(dim='time')
Q_REPE_AM = Q.loc[AM_REPE_idx]#.mean(dim='time')
DIVMF_REPE_AM = DIVMF.loc[AM_REPE_idx]#.mean(dim='time')

importlib.reload(figu)
fig, axes=pplt.subplots([[1, 4], [2, 5], [3, 6]], 
                        figsize=(10, 10), proj='cyl')

titles = ['SLP&uv850 EPCP', 'GPH&uv500 EPCP', 'q&uvmf EPCP', 
          'SLP&uv850 REPE', 'GPH&uv500 REPE', 'q&uvmf REPE',]
# units  = ['$Pa$', '$m^2 / s^2$', '$kg / kg$', 
#           '$Pa$', '$m^2 / s^2$', '$kg / kg$']
units  = ['', '', '', 
          '', '', '']
[geo_format(ax, titles[i], units[i]) for i, ax in enumerate(axes)]

scale1=1; scale2=0.75; scale3=40

ax1=axes[0]
figu.anomaly_plot2(ax1, lon, lat, SLP_EPCP_AM, SLP_AM, cmap='purple_orange', 
             levels=levels_slp, **cf_kwards)
figu.anomaly_wind_plot(ax1, lonu, latu, 
               U850_EPCP_AM, V850_EPCP_AM, 
               U850_AM, V850_AM, scale=scale1, **wind_kwards)

ax2=axes[1]
figu.anomaly_plot2(ax2, lon, lat, Z_EPCP_AM, Z_AM, cmap='anomaly', 
             levels=levels_z, **cf_kwards)
figu.anomaly_wind_plot(ax2, lonu, latu, 
               U500_EPCP_AM, V500_EPCP_AM, 
               U500_AM, V500_AM, scale=scale2, **wind_kwards)

ax3=axes[2]
figu.anomaly_plot2(ax3, lon, lat, Q_EPCP_AM, Q_AM, cmap='precip_diff', 
             levels=levels_q, **cf_kwards)
figu.anomaly_wind_plot(ax3, lonu, latu, 
               UMF_EPCP_AM, VMF_EPCP_AM, 
               UMF_AM, VMF_AM, scale=scale3, **wind_kwards)

ax1=axes[3]
cf1=figu.anomaly_plot2(ax1, lon, lat, SLP_REPE_AM, SLP_AM, cmap='purple_orange', 
             levels=levels_slp, **cf_kwards)
figu.anomaly_wind_plot(ax1, lonu, latu, 
               U850_REPE_AM, V850_REPE_AM, 
               U850_AM, V850_AM, scale=scale1, **wind_kwards)

ax2=axes[4]
cf2=figu.anomaly_plot2(ax2, lon, lat, Z_REPE_AM, Z_AM, cmap='anomaly', 
             levels=levels_z, **cf_kwards)
figu.anomaly_wind_plot(ax2, lonu, latu, 
               U500_REPE_AM, V500_REPE_AM, 
               U500_AM, V500_AM, scale=scale2, **wind_kwards)

ax3=axes[5]
cf3=figu.anomaly_plot2(ax3, lon, lat, Q_REPE_AM, Q_AM, cmap='precip_diff', 
             levels=levels_q, **cf_kwards)
figu.anomaly_wind_plot(ax3, lonu, latu, 
               UMF_REPE_AM, VMF_REPE_AM, 
               UMF_AM, VMF_AM, scale=scale3, **wind_kwards)

ax1.colorbar(cf1, loc='right', length=1, label='')
ax2.colorbar(cf2, loc='right', length=1, label='')
ax3.colorbar(cf3, loc='right', length=1, label='')

fig.savefig('./pics/FIG5_EPCPs_AM.png')
fig.show()


#%%
JJA_EPCP_idx = CNN_res.index[Cond_JJA & Cond_EPCP]
JJA_REPE_idx = CNN_res.index[Cond_JJA & Cond_REPE]
JJA_idx = CNN_res.index[Cond_JJA]

U850_JJA = U850.loc[JJA_idx].mean(dim='time')
V850_JJA = V850.loc[JJA_idx].mean(dim='time')
SLP_JJA = SLP.loc[JJA_idx].mean(dim='time')
U500_JJA = U500.loc[JJA_idx].mean(dim='time')
V500_JJA = V500.loc[JJA_idx].mean(dim='time')
Z_JJA = Z.loc[JJA_idx].mean(dim='time')
UMF_JJA = UMF.loc[JJA_idx].mean(dim='time')
VMF_JJA = VMF.loc[JJA_idx].mean(dim='time')
Q_JJA = Q.loc[JJA_idx].mean(dim='time')
DIVMF_JJA = DIVMF.loc[JJA_idx].mean(dim='time')

U850_EPCP_JJA = U850.loc[JJA_EPCP_idx]#.mean(dim='time')
V850_EPCP_JJA = V850.loc[JJA_EPCP_idx]#.mean(dim='time')
SLP_EPCP_JJA = SLP.loc[JJA_EPCP_idx]#.mean(dim='time')
U500_EPCP_JJA = U500.loc[JJA_EPCP_idx]#.mean(dim='time')
V500_EPCP_JJA = V500.loc[JJA_EPCP_idx]#.mean(dim='time')
Z_EPCP_JJA = Z.loc[JJA_EPCP_idx]#.mean(dim='time')
UMF_EPCP_JJA = UMF.loc[JJA_EPCP_idx]#.mean(dim='time')
VMF_EPCP_JJA = VMF.loc[JJA_EPCP_idx]#.mean(dim='time')
Q_EPCP_JJA = Q.loc[JJA_EPCP_idx]#.mean(dim='time')
DIVMF_EPCP_JJA = DIVMF.loc[JJA_EPCP_idx]#.mean(dim='time')

U850_REPE_JJA = U850.loc[JJA_REPE_idx]#.mean(dim='time')
V850_REPE_JJA = V850.loc[JJA_REPE_idx]#.mean(dim='time')
SLP_REPE_JJA = SLP.loc[JJA_REPE_idx]#.mean(dim='time')
U500_REPE_JJA = U500.loc[JJA_REPE_idx]#.mean(dim='time')
V500_REPE_JJA = V500.loc[JJA_REPE_idx]#.mean(dim='time')
Z_REPE_JJA = Z.loc[JJA_REPE_idx]#.mean(dim='time')
UMF_REPE_JJA = UMF.loc[JJA_REPE_idx]#.mean(dim='time')
VMF_REPE_JJA = VMF.loc[JJA_REPE_idx]#.mean(dim='time')
Q_REPE_JJA = Q.loc[JJA_REPE_idx]#.mean(dim='time')
DIVMF_REPE_JJA = DIVMF.loc[JJA_REPE_idx]#.mean(dim='time')

importlib.reload(figu)
fig, axes=pplt.subplots([[1, 4], [2, 5], [3, 6]], 
                        figsize=(10, 10), proj='cyl')

titles = ['SLP&uv850 EPCP', 'GPH&uv500 EPCP', 'q&uvmf EPCP', 
          'SLP&uv850 REPE', 'GPH&uv500 REPE', 'q&uvmf REPE',]
# units  = ['$Pa$', '$m^2 / s^2$', '$kg / kg$', 
#           '$Pa$', '$m^2 / s^2$', '$kg / kg$']
units  = ['', '', '', 
          '', '', '']
[geo_format(ax, titles[i], units[i]) for i, ax in enumerate(axes)]

scale1=1; scale2=0.75; scale3=40

ax1=axes[0]
figu.anomaly_plot2(ax1, lon, lat, SLP_EPCP_JJA, SLP_JJA, cmap='purple_orange', 
             levels=levels_slp, **cf_kwards)
figu.anomaly_wind_plot(ax1, lonu, latu, 
               U850_EPCP_JJA, V850_EPCP_JJA, 
               U850_JJA, V850_JJA, scale=scale1, **wind_kwards)

ax2=axes[1]
figu.anomaly_plot2(ax2, lon, lat, Z_EPCP_JJA, Z_JJA, cmap='anomaly', 
             levels=levels_z, **cf_kwards)
figu.anomaly_wind_plot(ax2, lonu, latu, 
               U500_EPCP_JJA, V500_EPCP_JJA, 
               U500_JJA, V500_JJA, scale=scale2, **wind_kwards)

ax3=axes[2]
figu.anomaly_plot2(ax3, lon, lat, Q_EPCP_JJA, Q_JJA, cmap='precip_diff', 
             levels=levels_q, **cf_kwards)
figu.anomaly_wind_plot(ax3, lonu, latu, 
               UMF_EPCP_JJA, VMF_EPCP_JJA, 
               UMF_JJA, VMF_JJA, scale=scale3, **wind_kwards)

ax1=axes[3]
cf1=figu.anomaly_plot2(ax1, lon, lat, SLP_REPE_JJA, SLP_JJA, cmap='purple_orange', 
             levels=levels_slp, **cf_kwards)
figu.anomaly_wind_plot(ax1, lonu, latu, 
               U850_REPE_JJA, V850_REPE_JJA, 
               U850_JJA, V850_JJA, scale=scale1, **wind_kwards)

ax2=axes[4]
cf2=figu.anomaly_plot2(ax2, lon, lat, Z_REPE_JJA, Z_JJA, cmap='anomaly', 
             levels=levels_z, **cf_kwards)
figu.anomaly_wind_plot(ax2, lonu, latu, 
               U500_REPE_JJA, V500_REPE_JJA, 
               U500_JJA, V500_JJA, scale=scale2, **wind_kwards)

ax3=axes[5]
cf3=figu.anomaly_plot2(ax3, lon, lat, Q_REPE_JJA, Q_JJA, cmap='precip_diff', 
             levels=levels_q, **cf_kwards)
figu.anomaly_wind_plot(ax3, lonu, latu, 
               UMF_REPE_JJA, VMF_REPE_JJA, 
               UMF_JJA, VMF_JJA, scale=scale3, **wind_kwards)

ax1.colorbar(cf1, loc='right', length=1, label='')
ax2.colorbar(cf2, loc='right', length=1, label='')
ax3.colorbar(cf3, loc='right', length=1, label='')

fig.savefig('./pics/FIG5_EPCPs_JJA.png')
fig.show()

#%%