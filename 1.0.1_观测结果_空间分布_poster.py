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
    .quantile(q=para.quantile, interpolation=para.interpolation)

ep_freq_mon = xr.where(prec>=prec_threshold, 1, 0).resample(time='M').sum()
ep_freq_annual_avg = ep_freq_mon.groupby(ep_freq_mon.time.dt.month).mean()\
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

proj_lcc = ccrs.LambertConformal(central_latitude=90, central_longitude=104)
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
    ax.colorbar(cf, loc='r', ticks=levels[::2], )#space=-fig.get_size_inches()[1]/16)

def clim_plot_inset(ax, data, levels, **kwards):
    kwards_update = figu.kw_update(kwards_cf, kwards)
    figu.contourf_plot(ax, lon, lat, data, levels=levels, 
                       transform=ccrs.PlateCarree(), 
                       **kwards_update)

month = np.arange(1, 13)
month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, axes = pplt.subplots([[1, 1, 1, 1, 2, 2, 2]], figsize=(12, 4), dpi=250, 
                          proj=['cart', proj_lcc])

axes[0].format(xticks=month, xminorticks=[], )
axes[0].bar(month[3:8], ep_freq_annual_avg.data[3:8], c='pink5', label='75%', zorder=2)
axes[0].bar(month, ep_freq_annual_avg.data, c='grey5', label='25%')
axes[0].format(ylabel='Frequency (days/month)', xlabel='Month', ylim=(0, 2.5))
axes.legend(ncol=1)
axes[0].set_title('Freq of EP', loc='left')

axes_inset = figu.geo_format_lambert(fig, axes[1:], **kwards_geo)
# 极端降水总量
levels  = np.linspace(100, 500, 9)
title  = 'EP'
unit  = 'mm/AMJJA'
clim_plot(axes[1], ep_amount_mean.data, levels, title, unit)
clim_plot_inset(axes_inset[0], ep_amount_mean.data, levels)

fig.savefig('./pics/FIG1_降水空间分布_poster.png', dpi=400)
fig.show()
#%%
