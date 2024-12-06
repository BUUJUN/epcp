# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/27 10:44:46 
 
@author: BUUJUN WANG
"""
#%%
import numpy as np
import xarray as xr
import proplot as pplt

import importlib
import buujun.figure_1d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)

#%%
prec = xr.open_dataset(para.CN05_path).pre\
    .sel(time=para.P_study, lon=para.lon_prec, lat=para.lat_prec)

#%%
prec_avg = prec.mean(dim=['lon', 'lat'])
prec_avg_threshold = xr.where(prec_avg>1, prec_avg, np.nan)\
    .quantile(q=para.quantile, interpolation=para.interpolation)

prec_avg_mon = prec_avg.resample(time='M').sum()
prec_avg_mon_annual = prec_avg_mon.groupby(prec_avg_mon.time.dt.month).mean()

ep_freq_avg_mon = xr.where(prec_avg>=prec_avg_threshold, 1, 0).resample(time='M').sum()
ep_freq_avg_mon_annual = ep_freq_avg_mon.groupby(ep_freq_avg_mon.time.dt.month).mean()


#%%
importlib.reload(figu)
month = np.arange(1, 13)
month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig, axes = pplt.subplots([[1, 2],], figsize=(10, 4), dpi=100)
axes.format(abcloc='ul', abc='a', xticks=month, xminorticks=[], )
axes[0].bar(month[3:8], prec_avg_mon_annual.data[3:8], c='pink5', label='67%', zorder=2)
axes[0].bar(month, prec_avg_mon_annual.data, c='grey5', label='33%')
axes[0].format(ylabel='Precipitation (mm/month)', xlabel='Month')
axes[1].bar(month[3:8], ep_freq_avg_mon_annual.data[3:8], c='pink5', label='75%', zorder=2)
axes[1].bar(month, ep_freq_avg_mon_annual.data, c='grey5', label='25%')
axes[1].format(ylabel='Frequency (days/month)', xlabel='Month')
axes.legend(ncol=1)

# %%
