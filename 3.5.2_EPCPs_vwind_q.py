# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/10/25 01:06:09 
 
@author: BUUJUN WANG
"""
#%%
import xarray as xr
import numpy as np
import pandas as pd
import proplot as pplt
import scipy.stats as stats
import importlib
import buujun.figure_1d as figu
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
    return array

#%% 全局参数
AM = [4, 5]
JJA = [6, 7, 8]
SC = dict(longitude=slice(108, 120), latitude=slice(26.5, 21))
SC_Sbound = dict(longitude=slice(108, 120), latitude=slice(21, 21))

CNN_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)
CNN_res = CNN_res.loc['1967-01-01':'2018-12-31']
v = read_data('v', level=850).sel(time=slice('1967', '2018'))\
    .sel(SC_Sbound).mean(dim=SC_Sbound.keys())
q = read_data('q').sel(time=slice('1967', '2018'))\
    .sel(SC).mean(dim=SC.keys())

year = np.asarray(list(set(CNN_res.index.year.values)), dtype='int')

kwards_l1=dict(lw=2, c='grey6', m='o')
kwards_l2=dict(lw=4, c='grey9')
kwards_l30=dict(lw=2, c='grey', ls='--')
kwards_l31=dict(lw=2, c='k', ls='--')

#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 4], [2, 5], [3, 6]], 
                          figsize=(12, 9))
axes.format(xloc='bottom', yloc='left', xlim=(1965, 2020))

CNN_res_AM = CNN_res[np.isin(CNN_res.index.month, AM)]
epcp_freq_AM = CNN_res_AM.predict_ep
epcp_freq_AM_yr = epcp_freq_AM.resample('Y').mean() * 30
epcp_freq_AM_yrr9 = calc.filtering(epcp_freq_AM_yr.copy())

ax1 = axes[0]
ax1.format(title='Freq of EPCPs in AM')
ax1.plot(year, epcp_freq_AM_yr.values, **kwards_l1)
ax1.plot(year, epcp_freq_AM_yrr9.values, **kwards_l2)
figu.trend_plot(ax1, year, epcp_freq_AM_yr.values, demean=False, **kwards_l31)

v_AM = v.to_series().loc[CNN_res_AM.index]
v_AM_yr = v_AM.resample('Y').mean()
v_AM_yrr9 = calc.filtering(v_AM_yr.copy())

ax1 = axes[1]
ax1.format(title='850-hPa v-wind in AM')
ax1.plot(year, v_AM_yr.values, **kwards_l1)
ax1.plot(year, v_AM_yrr9.values, **kwards_l2)
figu.trend_plot(ax1, year, v_AM_yr.values, demean=False, **kwards_l31)

q_AM = q.to_series().loc[CNN_res_AM.index][epcp_freq_AM==1]
q_AM_yr = q_AM.resample('Y').mean()
q_AM_yrr9 = calc.filtering(q_AM_yr.copy())

ax1 = axes[2]
ax1.format(title='Specific Humidity in AM')
ax1.plot(year, q_AM_yr.values, **kwards_l1)
ax1.plot(year, q_AM_yrr9.values, **kwards_l2)
figu.trend_plot(ax1, year, q_AM_yr.values, demean=False, **kwards_l30)

# fig.savefig('./pics/FIG8_EPCPs_V_Q_(AM).png')
# fig.show()

# importlib.reload(figu)
# fig, axes = pplt.subplots([[1], [2], [3]], figsize=(7, 10))
# axes.format(xloc='bottom', yloc='left', xlim=(1965, 2020))

CNN_res_JJA = CNN_res[np.isin(CNN_res.index.month, JJA)]
epcp_freq_JJA = CNN_res_JJA.predict_ep
epcp_freq_JJA_yr = epcp_freq_JJA.resample('Y').mean() * 30
epcp_freq_JJA_yrr9 = calc.filtering(epcp_freq_JJA_yr.copy())

ax1 = axes[3]
ax1.format(title='Freq of EPCPs in JJA')
ax1.plot(year, epcp_freq_JJA_yr.values, **kwards_l1)
ax1.plot(year, epcp_freq_JJA_yrr9.values, **kwards_l2)
figu.trend_plot(ax1, year, epcp_freq_JJA_yr.values, demean=False, **kwards_l30)

v_JJA = v.to_series().loc[CNN_res_JJA.index]
v_JJA_yr = v_JJA.resample('Y').mean()
v_JJA_yrr9 = calc.filtering(v_JJA_yr.copy())

ax1 = axes[4]
ax1.format(title='850-hPa v-wind in JJA')
ax1.plot(year, v_JJA_yr.values, **kwards_l1)
ax1.plot(year, v_JJA_yrr9.values, **kwards_l2)
figu.trend_plot(ax1, year, v_JJA_yr.values, demean=False, **kwards_l30)

q_JJA = q.to_series().loc[CNN_res_JJA.index][epcp_freq_JJA==1]
q_JJA_yr = q_JJA.resample('Y').mean()
q_JJA_yrr9 = calc.filtering(q_JJA_yr.copy())

ax1 = axes[5]
ax1.format(title='Specific Humidity in JJA')
ax1.plot(year, q_JJA_yr.values, **kwards_l1)
ax1.plot(year, q_JJA_yrr9.values, **kwards_l2)
figu.trend_plot(ax1, year, q_JJA_yr.values, demean=False, **kwards_l31)

fig.savefig('./pics/FIG8_EPCPs_V_Q.png')
fig.show()

# %%
# 对 JJA EPCPs 滑动T检验
importlib.reload(figu)
fig, axes = pplt.subplots([[1], [2]], figsize=(7, 7))
axes.format(xloc='bottom', yloc='left', xlim=(1965, 2020))

epcp_freq_JJA_yr_detrend = calc.detrend(epcp_freq_JJA_yr)
v_JJA_yr_detrend = calc.detrend(v_JJA_yr)

ax1 = axes[0]
ax1.format(title='Moving T-test for Freq of EPCPs in JJA')
figu.rolling_ttest_plot(ax1, year, epcp_freq_JJA_yr_detrend, alpha=0.05)

ax1 = axes[1]
ax1.format(title='Moving T-test for 850-hPa v-wind in JJA')
figu.rolling_ttest_plot(ax1, year, v_JJA_yr_detrend, alpha=0.05)

fig.savefig('./pics/FIGS9_MT-test_EPCPs_V(JJA).png')
fig.show()

# %%
