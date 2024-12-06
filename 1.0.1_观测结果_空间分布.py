# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/03/07 15:23:26 
 
@author: BUUJUN WANG

FIG 1
(a) 各月降水量
(b) 各月极端事件频次
(c) 降水量气候态
(d) P90 空间分布
(e) 极端降水量气候态
"""
#%%
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=r'.*deprecat.*')

import numpy as np
import xarray as xr
import proplot as pplt
import cartopy.crs as ccrs

import importlib
import buujun.figure_2d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)

# %%
prec = xr.open_dataset(para.CN05_path).pre\
    .sel(time=para.P_study, lon=para.lon_prec, lat=para.lat_prec)

prec_avg = prec.mean(dim=['lon', 'lat'])
prec_avg_mon = prec_avg.resample(time='M').sum()
prec_avg_annual = prec_avg_mon.groupby(prec_avg_mon.time.dt.month).mean()

prec_threshold = xr.where(prec>1, prec, np.nan)\
    .quantile(q=para.quantile, method=para.interpolation)

ep_freq_mon = para.fill_mask(xr.where(prec>=prec_threshold, 1, 0).resample(time='M').sum())
ep_freq_annual_avg = ep_freq_mon.groupby(ep_freq_mon.time.dt.month).mean()\
    .mean(dim=['lon', 'lat'])

ep_prec_mon = para.fill_mask(prec.where(prec>=prec_threshold).resample(time='M').sum())
ep_prec_annual_avg = ep_prec_mon.groupby(ep_prec_mon.time.dt.month).mean()\
    .mean(dim=['lon', 'lat'])

#%%
prec_mask = ~np.isnan(xr.open_dataset(para.prec_china_path).prec_threshold)
prec_amount_mean = xr.open_dataset(para.prec_china_path).prec\
    .resample(time='Y').sum().mean(dim='time').where(prec_mask, np.nan)

ep_amount = xr.open_dataset(para.prec_china_path).ep_amount
ep_amount_mean = ep_amount.mean(dim='year')
# ep_amount_trend, _, _, pvalue, _ = calc.linregress_n(ep_amount, dim='year')
# ep_amount_diff, pvalue_diff = calc.diff_n(data_1=ep_amount.loc[:, :, para.PE], 
#                                           data_2=ep_amount.loc[:, :, para.PL], 
#                                           axis=2)

lon = ep_amount.lon
lat = ep_amount.lat

lon_start = para.lon_prec.start
lon_stop = para.lon_prec.stop
lat_start = para.lat_prec.start
lat_stop= para.lat_prec.stop

# %%
importlib.reload(figu)

proj_lcc = pplt.Proj('lcc', lon_0=104, lat_0=90)

kwards_geo = dict(lonlat=[77, 131, 15, 55], 
                  lonticks=np.arange(70, 135, 20), 
                  latticks=np.arange(10, 60, 10), 
                  coastline=False)
kwards_cf = dict(cmap='wet', extend='both')
kwards_test = dict(transform=ccrs.PlateCarree(), zorder=2, colors='None', hatches=['..'], )

def clim_plot(ax, data, levels, title, unit, **kwards):
    kwards_update = figu.kw_update(kwards_cf, kwards)
    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    n_p = 5
    lon_grid_p = np.linspace(lon_start, lon_stop, n_p)
    lon_grid = np.concatenate([lon_grid_p, lon_grid_p[::-1], [lon_start]])
    lat_grid = np.concatenate([np.full(n_p, lat_start), np.full(n_p, lat_stop), [lat_start]])
    ax.plot(lon_grid, lat_grid, 
            color='r', lw=1.5, transform=ccrs.PlateCarree())
    cf = figu.contourf_plot(ax, lon, lat, data, 
                            levels=levels, transform=ccrs.PlateCarree(), 
                            **kwards_update)
    ax.colorbar(cf, ticks=levels, space=-fig.get_size_inches()[1]/4)

def clim_plot_inset(ax, data, levels, **kwards):
    kwards_update = figu.kw_update(kwards_cf, kwards)
    figu.contourf_plot(ax, lon, lat, data, levels=levels, 
                       transform=ccrs.PlateCarree(), 
                       **kwards_update)

month = np.arange(1, 13)
month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, axes = pplt.subplots(array=[[0, 1, 1, 1, 0, 2, 2, 2, 0],
                                 [3, 3, 3, 4, 4, 4, 5, 5, 5], 
                                 [3, 3, 3, 4, 4, 4, 5, 5, 5]], 
                          figsize=(12, 7), dpi=250, 
                          proj=['cart', 'cart', 
                                proj_lcc, proj_lcc, proj_lcc], )

prec_sum=prec_avg_annual.data[3:8]
prec_anu=prec_avg_annual.data
fepe_sum=ep_freq_annual_avg.data[3:8]
fepe_anu=ep_freq_annual_avg.data
pepe_sum=ep_prec_annual_avg.data[3:8]
pepe_anu=ep_prec_annual_avg.data

prec_sum_prop = prec_sum.sum() / prec_anu.sum()
fepe_sum_prop = fepe_sum.sum() / fepe_anu.sum()
pepe_sum_prop = pepe_sum.sum() / pepe_anu.sum()

axes[0].bar(month[3:8], prec_sum, 
            c='pink5', label=f'{prec_sum_prop:.0%}', zorder=2)
axes[0].bar(month, prec_anu, 
            c='grey5', label=f'{1-prec_sum_prop:.0%}')

axes[1].bar(month[3:8], pepe_sum, 
            c='pink5', label=f'{pepe_sum_prop:.0%}', zorder=2)
axes[1].bar(month, pepe_anu, 
            c='grey5', label=f'{1-pepe_sum_prop:.0%}')

axes[0:2].legend(ncol=1)
axes[0:2].format(xticks=month, xminorticks=[], )
axes[0].format(ylim=(0, 300), ylabel='', xlabel='Month', )
axes[1].format(ylim=(0, 150), ylabel='', xlabel='Month', )
axes[0].set_title('Precipitation', loc='left')
axes[1].set_title('Extreme Prec', loc='left')

axes_inset = figu.geo_format_lambert(fig, axes[2:], **kwards_geo)

levels  = np.linspace(200, 1400, 7)
title  = 'Precipitation'
unit  = ''
clim_plot(axes[2], prec_amount_mean.data, levels, title, unit)
clim_plot_inset(axes_inset[0], prec_amount_mean.data, levels)

# 95分位数
prec_threshold_china = xr.open_dataset(para.prec_china_path).prec_threshold
levels  = np.linspace(5, 35, 7)
title  = 'P90'
unit  = ''
clim_plot(axes[3], prec_threshold_china.data, levels, title, unit)
clim_plot_inset(axes_inset[1], prec_threshold_china.data, levels)

# 极端降水总量
levels  = np.linspace(120, 480, 7)
title  = 'Extreme Prec'
unit  = ''
clim_plot(axes[4], ep_amount_mean.data, levels, title, unit)
clim_plot_inset(axes_inset[2], ep_amount_mean.data, levels)

fig.savefig('./pics/FIG1_降水空间分布.png', dpi=400)
fig.show()
 #%%
