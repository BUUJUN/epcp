# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/03/07 15:51:37 
 
@author: BUUJUN WANG
"""

#%%
import importlib
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(para)
importlib.reload(calc)

# configuring proplot
from buujun import figure_init
importlib.reload(figure_init)

#%%
# colors
cmap_idx = np.power(np.linspace(0.25, 0.9, 6), 1)
reds = pplt.Colormap('reds2')(cmap_idx)
blues = pplt.Colormap('blues2')(cmap_idx)
greens = pplt.Colormap('greens')(cmap_idx)
oranges = pplt.Colormap('oranges')(cmap_idx)
browns = pplt.Colormap('browns1')(cmap_idx)
purples = pplt.Colormap('purples')(cmap_idx)
white = np.ones(4).reshape(1,-1)  # [[1,1,1,1]]
twilight_lp = pplt.Colormap('twilight')(np.linspace(0, 0.5, 21))
twilight_rp = pplt.Colormap('twilight')(np.linspace(0.5, 1, 21))
dense_rp = pplt.Colormap('dense')(np.linspace(0.5, 1, 21))
rocket = pplt.Colormap('rocket')(np.linspace(0, 0.7, 15))

cmap_rocket = pplt.Colormap(rocket, name='rocket_p')
cmap_twilight_lp = pplt.Colormap(twilight_lp, name='twilight_lp')
cmap_dense_rp = pplt.Colormap(dense_rp, name='dense_rp')
cmap_test = pplt.Colormap(['#000000'], name='test')

#%%
# cmaps
concat = lambda colors1, colors2: np.concatenate([colors1[::-1], white, colors2])
cmap_blue_red = pplt.Colormap(concat(blues, reds), name='blue_red')
cmap_brown_green = pplt.Colormap(concat(browns, greens), name='brown_green')
cmap_purple_orange = pplt.Colormap(concat(purples, oranges), name='purple_orange')


cmap_precip_level = pplt.Colormap(
    ['#EAF7FD', '#81C1E7', '#488FC5', '#49A96C', '#79C24C', '#AAD152', 
     '#F0E65B', '#F8BC4A', '#F79A3B', '#E54829', '#9B171D'], name='precip_level')

cmap_wet = pplt.Colormap(
    ['#EAF7FD', '#C8E9C3', '#A3DBB6', '#76C9C5', 
     '#4AAED0', '#2687BB', '#0863A6', '#084B8D'], name='wet')

cmap_precip_diff = pplt.Colormap(
    ['#B76A23', '#CE863D', '#E2A764', '#F5CE85', '#F5E19F', '#F8EBBF', '#FFFFFF', 
     '#99FFFF', '#65FFFF', '#00FFFF', '#00CDCD', '#009A9A', '#006666'], name='precip_diff')

cmap_precip_diff = pplt.Colormap(
    ['#B76A23', '#CE863D', '#E2A764', '#F5CE85', '#F5E19F', '#FFFFFF', 
     '#65FFFF', '#00FFFF', '#00CDCD', '#009A9A', '#006666'], name='precip_diff1')

cmap_precip_diff2 = pplt.Colormap(
    ['#B76A23', '#CE863D', '#E2A764', '#F5CE85', '#F5E19F', '#FFFFFF', 
     '#9AF0B4', '#2EA897', '#0071B2', '#014F8D', '#023E82'], name='precip_diff2')

cmap_precip_diff3 = pplt.Colormap(
    ['#B76A23', '#CE863D', '#E2A764', '#F5CE85', '#F5E19F', '#FFFFFF', 
     '#A9DEB6', '#4DB4D4', '#278DBF', '#0268AE', '#023E82'], name='precip_diff3')

cmap_precip_diff4 = pplt.Colormap(
    ['#B76A23', '#CE863D', '#E2A764', '#F5CE85', '#F5E19F',
     '#A9DEB6', '#4DB4D4', '#278DBF', '#0268AE', '#023E82'], name='precip_diff4')

cmap_anomalyw = pplt.Colormap(
    ['#003A7A', '#005EB0', '#0081E6', '#2C91ED', '#59A1F5', 
     '#74B3F8', '#91C3FE', '#AED3FF', '#CDE2FF', '#FFFFFF',
     '#FEFDBC', '#FFDE99', '#FEBE76', '#FF9F55', '#FF7F34', 
     '#FF5C1B', '#FF3801', '#CD290D', '#981B19', ], name='anomalyw')

cmap_anomaly = pplt.Colormap(
    ['#003A7A', '#0058A8', '#0076D5', '#1E8AEA', '#3397F6',
     '#5FA5F6', '#78B5F9', '#91C3FE', '#AAD1FF', '#C4DDFF', 
     '#D8E9FE', '#FDF7CD', 
     '#FDF4B4', '#FFD992', '#FEBE76', '#FFA459', '#FF873B',
     '#FF6B23', '#FF4E0E', '#F03306', '#C52711', '#981B19'], name='anomaly')

cmap_anomaly2 = pplt.Colormap(
    ['#194EFF', '#1960FF', '#199FFF', '#19C0FF', '#00D0FF', '#25FFFF', 
     '#53FFFF', '#7FFFFF', '#FFFFFF', '#FFF000', '#FFE22E', 
     '#FFAC00', '#FF6E00', '#FF4600', '#FF0000', '#C90000', '#9F0000'], name='anomaly2'
)
cmap_anomaly2w = pplt.Colormap(
    ['#194EFF', '#1960FF', '#199FFF', '#19C0FF', '#00D0FF', '#25FFFF', 
     '#53FFFF', '#7FFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFF000', '#FFE22E', 
     '#FFAC00', '#FF6E00', '#FF4600', '#FF0000', '#C90000', '#9F0000'], name='anomaly2w'
)


#%%
# contents
lonlat = [75, 135, 0, 40]
proj_cyl = ccrs.PlateCarree()
# kwards
kwards_coast = dict(edgecolor='k', lw=1.5, zorder=1)
kwards_country = dict(edgecolor='k', lw=1.5, zorder=1)
kwards_ticks = dict(top=True, right=True, bottom=True, left=True)
kwards_majorticks = dict(length=pplt.rc['tick.len'], width=pplt.rc['tick.width'], pad=1)
kwards_minorticks = dict(length=pplt.rc['tick.len']*pplt.rc['tick.lenratio'], 
                         width=pplt.rc['tick.width']*pplt.rc['tick.widthratio'])
kwards_cf = dict(transform=proj_cyl, zorder=0,)
# kwards_test = dict(transform=proj_cyl, zorder=2, colors='None', hatches=['//'], )
kwards_test = dict(transform=proj_cyl, zorder=2, colors='None', hatches=['.'], )
kwards_test_scatter = dict(transform=proj_cyl, 
                           zorder=2, 
                           marker='.', 
                           cmap='test', alpha=1)
kwards_wind_old = dict(transform=proj_cyl, zorder=3, pivot='mid', color='k', 
                   scale=1, scale_units='xy', width=pplt.rc['figure.dpi']/25, units='dots', 
                   headwidth=3, headlength=5, headaxislength=4)

kwards_wind = dict(transform=proj_cyl, zorder=3, pivot='mid', color='k', 
                   scale=None, ## 根据数据来的，0.2 m/s 设为 5
                   scale_units='inches', units='dots', 
                   width=pplt.rc['figure.dpi']/16, 
                   headwidth=3, headlength=5, headaxislength=4.5)

def kw_update(kwards_old, kwards_new):
    kwards = kwards_old.copy()
    kwards.update(kwards_new)
    return kwards


def add_feature_china(axes):
    import geopandas as gpd
    shp_china_country = gpd.read_file('/home/yangsong3/wangbj/map-datas/china/china_country.shp')
    shp_china_nine_dotted = gpd.read_file('/home/yangsong3/wangbj/map-datas/china/china_nine_dotted_line.shp')
    cf_china_country = cfeature.ShapelyFeature(shp_china_country.geometry, ccrs.PlateCarree())
    cf_china_nine_dotted = cfeature.ShapelyFeature(shp_china_nine_dotted.geometry, ccrs.PlateCarree())
    axes.add_feature(cf_china_country, linewidth=1.5, edgecolor='k', facecolor='None', zorder=1)
    axes.add_feature(cf_china_nine_dotted, linewidth=1.5, edgecolor='k', facecolor='None', zorder=1)
    ax_inset = axes.inset([0.775, 0.02, 0.25, 0.25], transform='axes')
    ax_inset.set_extent([102, 123, 3, 27], ccrs.PlateCarree())
    ax_inset.add_feature(cf_china_country, linewidth=1, edgecolor='k', facecolor='None', zorder=1)
    ax_inset.add_feature(cf_china_nine_dotted, linewidth=1, edgecolor='k', facecolor='None', zorder=1)
    return ax_inset


#%%
def set_ticks(ax, lonticks, latticks):
    ax.set_xticks(lonticks, crs=proj_cyl)
    ax.set_yticks(latticks, crs=proj_cyl)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(which='major', **kwards_majorticks, **kwards_ticks)
    ax.tick_params(which='minor', **kwards_minorticks, **kwards_ticks)


def geo_format(ax, lonlat=lonlat, lonticks=None, latticks=None, coastline=True):
    if lonticks is None: lonticks=np.round(np.linspace(lonlat[0], lonlat[1], 6))
    if latticks is None: latticks=np.round(np.linspace(lonlat[2], lonlat[3], 5))
    ax.set_extent(lonlat, crs=proj_cyl)
    set_ticks(ax, lonticks, latticks)
    if coastline == True:
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), **kwards_coast)


def geo_format_lambert(fig, axes, lonlat=lonlat, lonticks=None, latticks=None, 
                       lonticks_minor=None, latticks_minor=None, 
                       coastline=True, china_border=True):
    
    from buujun import figure_lambert_ticks as set_lt
    importlib.reload(set_lt)

    if lonticks is None: lonticks=np.round(np.linspace(lonlat[0], lonlat[1], 6))
    if latticks is None: latticks=np.round(np.linspace(lonlat[2], lonlat[3], 5))
    if lonticks_minor is None: lonticks_minor=np.linspace(lonticks[0], lonticks[-1], lonticks.size*3+1)
    if latticks_minor is None: latticks_minor=np.linspace(latticks[0], latticks[-1], latticks.size*4+1)

    if type(axes) == pplt.gridspec.SubplotGrid:
        for ax in axes: ax.set_extent(lonlat, crs=proj_cyl)
    else: axes.set_extent(lonlat, crs=proj_cyl)
    axes.format(grid=True, labels=False, 
                lonlocator=lonticks, latlocator=latticks, 
                coast=coastline, reso='hi')
    fig.canvas.draw()

    def set_lambert_ticks(ax):
        set_lt.lambert_ticks(ax, ticks=lonticks, tick_position='bottom')
        set_lt.lambert_ticks(ax, ticks=latticks, tick_position='left')
        set_lt.lambert_ticks(ax, ticks=lonticks_minor, tick_position='bottom', which='minor')
        set_lt.lambert_ticks(ax, ticks=latticks_minor, tick_position='left', which='minor')        

    if type(axes) == pplt.gridspec.SubplotGrid: 
        for ax in axes: set_lambert_ticks(ax)
    else: set_lambert_ticks(axes)

    axes.tick_params(which='major', bottom=True, left=True, right=False, top=False, **kwards_majorticks)
    axes.tick_params(which='minor', bottom=True, left=True, right=False, top=False, **kwards_minorticks)

    if china_border == True: 
        return add_feature_china(axes)


def contourf_plot(ax, lon, lat, data, cb=False, cbloc='b', **kwards):
    cf = ax.contourf(lon, lat, data, **kw_update(kwards_cf, kwards))
    if cb == True: ax.colorbar(cf, loc=cbloc, length=1)
    return cf

def contour_plot(ax, lon, lat, data, **kwards):
    cf = ax.contour(lon, lat, data, **kw_update(kwards_cf, kwards))
    return cf


def test_plot(ax, lon, lat, pvalue, alpha=0.05):
    # ax.contourf(lon, lat, np.where(pvalue<=alpha, 1, np.nan), **kwards_test)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    ax.scatter(lon_mesh, lat_mesh, 
               c=np.where(pvalue<=alpha, 1, np.nan), 
               **kwards_test_scatter)


def wind_plot_old(ax, lon, lat, u, v, 
              qkey=False, qscale=None, 
              qx=0.8, qy=None, qunit='$m/s$', **kwards):
    kwards_update = kw_update(kwards_wind_old, kwards)
    quiver = ax.quiver(lon, lat, u, v, **kwards_update)
    if qkey == True:
        if qscale is None: qscale = round(kwards_update['scale']*3, 2)
        if quivery is None: quivery = 1+ax.get_position().height/20
        ax.quiverkey(quiver, qx, qy, qscale, f'{qscale} '+qunit, 
                     labelpos='E', coordinates='axes', )
    return quiver


def wind_plot(ax, lon, lat, u, v, 
              qkey=False, qscale=None, 
              qx=0.8, qy=None, qunit='$m/s$', **kwards):
    kwards_update = kw_update(kwards_wind, kwards)
    qplot = ax.quiver(lon, lat, u, v, **kwards_update)
    if qkey == True:
        if qscale is None: qscale = qplot.scale
        if qy is None: qy = 1+ax.get_position().height/20
        ax.quiverkey(qplot, qx, qy, qscale, f'{qscale/qplot.scale:.1f} '+qunit, 
                     labelpos='E', coordinates='axes', 
                     fontproperties=dict(size=pplt.rc['font.large']))
    return qplot


def anomaly_plot(ax, lon, lat, data_part, data_clim, 
                 axis=0, cb=False, cbloc='b', alpha=0.05, **kwards):
    anomaly, pvalue = calc.anomaly_n(data_part=data_part, data_clim=data_clim, axis=axis)
    cf = contourf_plot(ax, lon, lat, anomaly, cb=cb, cbloc=cbloc, **kwards)
    test_plot(ax, lon, lat, pvalue, alpha=alpha)
    return cf


def anomaly_plot2(ax, lon, lat, data_part, data_clim, 
                 axis=0, cb=False, cbloc='b', alpha=0.05, **kwards):
    anomaly, pvalue = calc.anomaly_n(data_part=data_part, data_clim=data_clim, axis=axis)
    cf = contourf_plot(ax, lon, lat, np.where(pvalue<=alpha, anomaly, np.nan), cb=cb, cbloc=cbloc, **kwards)
    return cf


def anomaly_wind_plot(ax, lon, lat, u_part, v_part, u_clim, v_clim, 
                      axis=0, alpha=0.05, qkey=False, qscale=None, 
                      qx=0.8, qy=None, qunit='$m/s$', **kwards):
    anomaly_u, pvalue_u = calc.anomaly_n(u_part, u_clim, axis=axis)
    anomaly_v, pvalue_v = calc.anomaly_n(v_part, v_clim, axis=axis)
    pvalue = np.where(np.less(pvalue_u, pvalue_v), pvalue_u, pvalue_v) # 谁小取谁
    sig_u = np.where(pvalue<=alpha, anomaly_u, np.nan)
    sig_v = np.where(pvalue<=alpha, anomaly_v, np.nan)
    nsig_u = np.where(pvalue<=alpha, np.nan, anomaly_u)
    nsig_v = np.where(pvalue<=alpha, np.nan, anomaly_v)
    quiver = wind_plot(ax, lon, lat, sig_u, sig_v, color='k', 
                       qkey=qkey, qscale=qscale, 
                       qx=qx, qy=qy, qunit=qunit, **kwards)
    wind_plot(ax, lon, lat, nsig_u, nsig_v, color='grey', **kwards)
    return quiver


def diff_plot(ax, lon, lat, data_1, data_2,   # data_2 - data_1
              axis=0, cb=False, cbloc='b', alpha=0.05, **kwards):
    diff, pvalue = calc.diff_n(data_1, data_2, axis=axis)  # data_2 - data_1
    cf = contourf_plot(ax, lon, lat, diff, cb=cb, cbloc=cbloc, **kwards)
    test_plot(ax, lon, lat, pvalue, alpha=alpha)
    return cf


def diff_wind_plot(ax, lon, lat, u_1, v_1, u_2, v_2, 
                      axis=0, alpha=0.05, qkey=False, qscale=None, 
                      qx=0.8, qy=None, qunit='$m/s$', **kwards):
    diff_u, pvalue_u = calc.diff_n(u_1, u_2, axis=axis)
    diff_v, pvalue_v = calc.diff_n(v_1, v_2, axis=axis)
    pvalue = np.where(np.less(pvalue_u, pvalue_v), pvalue_u, pvalue_v) # 谁小取谁
    sig_u = np.where(pvalue<=alpha, diff_u, np.nan)
    sig_v = np.where(pvalue<=alpha, diff_v, np.nan)
    nsig_u = np.where(pvalue<=alpha, np.nan, diff_u)
    nsig_v = np.where(pvalue<=alpha, np.nan, diff_v)
    quiver = wind_plot(ax, lon, lat, sig_u, sig_v, color='k', 
                       qkey=qkey, qscale=qscale, 
                       qx=qx, qy=qy, qunit=qunit, **kwards)
    wind_plot(ax, lon, lat, nsig_u, nsig_v, color='grey9', **kwards)
    return quiver

def trend_plot(ax, lon, lat, data, 
               axis=0, cb=False, cbloc='b', alpha=0.05, **kwards):
    trend, _, _, pvalue, _ = calc.linregress_n(data, dim=data.dims[axis])
    cf = contourf_plot(ax, lon, lat, trend.data, cb=cb, cbloc=cbloc, **kwards)
    test_plot(ax, lon, lat, pvalue, alpha=alpha)
    return cf

def trend_wind_plot(ax, lon, lat, u, v, 
                    axis=0, alpha=0.05, qkey=False, qscale=None, 
                    qx=0.8, qy=None, qunit='$m/s$', **kwards):
    trend_u, _, _, pvalue_u, _ = calc.linregress_n(u, dim=u.dims[axis])
    trend_v, _, _, pvalue_v, _ = calc.linregress_n(v, dim=v.dims[axis])
    pvalue = np.where(np.less(pvalue_u, pvalue_v), pvalue_u, pvalue_v) # 谁小取谁
    sig_u = np.where(pvalue<=alpha, trend_u, np.nan)
    sig_v = np.where(pvalue<=alpha, trend_v, np.nan)
    nsig_u = np.where(pvalue<=alpha, np.nan, trend_u)
    nsig_v = np.where(pvalue<=alpha, np.nan, trend_v)
    quiver = wind_plot(ax, lon, lat, sig_u, sig_v, color='k', 
                       qkey=qkey, qscale=qscale, 
                       qx=qx, qy=qy, qunit=qunit, **kwards)
    wind_plot(ax, lon, lat, nsig_u, nsig_v, color='grey9', **kwards)
    return quiver
# %%
