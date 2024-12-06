# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2022/10/09 16:30:40 
 
@author: 王伯俊
"""
import numpy as np
import xarray as xr
import proplot as pplt

# 画图填充 360 度
def add_cyclic_lon_point(data_array, lon_name = 'lon'):
    """
    填充画图所需的经度 360° 的数据
    
    Parameters
    ----------
    data_array : xr.DataArray
    lon_name : data_array 的经度名称
    
    Returns
    -------
    data_array_wrap : 填充后的数据
    
    Notes
    -----
    
    """
    # from cartopy.util import add_cyclic_point
    # data_array_wrap, lon_wrap = add_cyclic_point(data_array.data, coord=data_array.coords[lon_name], axis=data_array.dims.index(lon_name))
    data_array_wrap = np.concatenate([data_array.data, data_array[{lon_name:[0]}].data], axis=data_array.dims.index(lon_name))
    lon_wrap = np.concatenate([data_array.coords[lon_name].data, [360]])
    data_array_wrap = data_array.loc[{lon_name:0}].expand_dims(dim={lon_name:lon_wrap}, axis=data_array.dims.index(lon_name)).assign_coords(coords={lon_name:lon_wrap.data}).copy(data=data_array_wrap.data)
    data_array_wrap.coords[lon_name].attrs = data_array.coords[lon_name].attrs
    return data_array_wrap

def kw_update(kwards_old, kwards_new):
    kwards = kwards_old.copy()
    kwards.update(kwards_new)
    return kwards

thetatick_kw_init = dict(lw=1, color='k')
gridline_kw_init = dict(lw=1, linestyle='--', facecolor='None', alpha=1, )

def set_axis(ax, thetaticks, thetaticklabels,
             rticks, rticklabels, ticklabelpad=10, 
             rmax = 1.5, thetaminorticks=None, thetatickscale=1, 
             **thetatick_kw):
    """
    设置泰勒图的角度轴
    
    Parameters
    ----------
    ax : 画图对象 PolarAxesSubplot
    thetaticks: 角度轴刻度
    thetaticklabels: 角都轴刻度标签
    rticks: 半径轴刻度
    rticklabels: 半径轴标签
    ticklabelpad: 标签到轴的空隙

    Returns
    -------
    None
    
    Notes
    -----
    
    """

    thetatick_kw = kw_update(thetatick_kw_init, thetatick_kw)
    ax.set_thetalim(thetamin=0, thetamax=90)
    ax.set_rlim(rmin=0, rmax=rmax)
    ax.grid(False)
    
    ax.set_thetagrids(
        angles=np.rad2deg(np.arccos(thetaticks)), 
        labels=thetaticklabels,
    )
    ax.set_rgrids(radii=rticks, labels=rticklabels)
    ax.tick_params(pad=ticklabelpad)
    rticklabelpad = 0.004*ticklabelpad
    rlabelpad = rticklabelpad * 5
    for i, r in enumerate(rticks):
        if r == 0: continue
        else: ax.text(
            np.arctan(rticklabelpad/r)+np.pi/2, 
            np.sqrt(r**2+rticklabelpad**2),
            s=str(rticks[i]), 
            ha='right', va='center', 
            fontsize=pplt.rc['tick.labelsize'])

    rlabel = 'Standardized Deviations'
    ax.text(np.arctan(rlabelpad/(rmax/2))+np.pi/2, 
            np.sqrt((rmax/2)**2+rlabelpad**2), 
            s=rlabel, ha='right', va='center', rotation='vertical', 
            fontsize=pplt.rc['leftlabel.size'])

    thetalabel = 'Correlation'
    ax.text(np.pi/4, rmax+rlabelpad, 
            s=thetalabel, ha='center', va='center', 
            rotation=-45,
            fontsize=pplt.rc['leftlabel.size'])
    
    thetatickscale *= 0.02

    if thetaminorticks is not None:
        rstart, rend = rmax*(1-thetatickscale/2), rmax
        for theta in np.arccos(thetaminorticks): 
            ax.plot([theta, theta], [rstart, rend], **thetatick_kw)

       
    rstart, rend = rmax*(1-thetatickscale), rmax
    for theta in np.arccos(thetaticks): 
        ax.plot([theta, theta], [rstart, rend], **thetatick_kw)

    return None
    

def set_thetaaxis(ax, ticks, labels, ticks_minor=None, scale=True, linewidth=1, linelength=1, color='k', pad=10):
    """
    设置泰勒图的角度轴
    
    Parameters
    ----------
    ax : 画图对象 PolarAxesSubplot
    ticks : theta 轴的 ticks, 单位是度
    labels : theta 轴的 labels
    ticks_minor : theta 轴的 minor_ticks, 单位是度
    scale : 是否绘制刻度
    linewidth : 刻度的粗细
    linelength : 刻度的长度
    color : theta 轴的 labels 和 刻度的颜色
    pad : labels 和 axis 之间的空白

    Returns
    -------
    ticklines objects
    
    Notes
    -----
    
    """
    ax.set_thetalim(thetamin=0, thetamax=90)
    ax.set_thetagrids(angles=ticks, labels=labels, color=color)
    ax.tick_params(axis='x', pad=pad)
    scale_length = linelength*0.02
    if ticks_minor is not None: 
        r_coordinate = [ax.get_rmax(), ax.get_rmax() * (1 - scale_length/2)]
        for theta in np.deg2rad(ticks_minor): 
            ax.plot([theta, theta], r_coordinate, lw=linewidth, color=color)
    if scale:
        r_coordinate = [ax.get_rmax(), ax.get_rmax() * (1 - scale_length)]
        for theta in np.deg2rad(ticks): 
            ax.plot([theta, theta], r_coordinate, lw=linewidth, color=color)
    return ax.xaxis.get_ticklines(), ax.xaxis.get_ticklabels()    

def set_raxis(ax, ticks, labels, color='k', pad=10):
    """
    设置泰勒图的半径轴
    
    Parameters
    ----------
    ax : 画图对象 PolarAxesSubplot
    ticks : r 轴的 ticks
    labels : r 轴的 labels
    pad : labels 和 axis 之间的空白

    Returns
    -------
    ticklines objects
    
    Notes
    -----
    
    """
    from numpy import pi
    import numpy as np
    ax.set_rlim(rmin=0, rmax=1.5)
    ax.set_rgrids(radii=ticks, labels=[], color=color)
    label_pad = ax.get_rmax()*0.002*pad
    for i, r in enumerate(ticks):
        if r == 0:
            ax.text(pi, label_pad, s=str(labels[i]), ha='right', va='center')
        else:
            ax.text(np.arctan(-label_pad/r), np.sqrt(r**2+label_pad**2), s=str(labels[i]), ha='center', va='top')
            ax.text(np.arctan(label_pad/r)+0.5*pi, np.sqrt(r**2+label_pad**2), s=str(ticks[i]), ha='right', va='center')
    return ax.yaxis.get_ticklines(), ax.yaxis.get_ticklabels()

def set_gridlines(
    ax, linewidth=1, alpha=1, 
    ticks_corr=None, grid_corr=False, color_corr='tab:gray', 
    ticks_std=None,  grid_std=False,  color_std='k', 
    ticks_bias=None, grid_bias=False, color_bias='lightpink', 
    **gridline_kw
):
    """
    设置 泰勒图的 网格线

    Parameters
    ----------
    ax : 画图对象 PolarAxesSubplot
    linewidth : 网格线的粗细
    alpha : 网格线的透明度
    ticks_corr : theta 轴需要添加网格的 ticks
    grid_corr : 如果没有给定 ticks, 是否默认所有的 ticks 都画出网格
    color_corr : 网格线的颜色

    Returns
    -------
    return_description

    Notes
    -----

    """
    import matplotlib.pyplot as plt
    gridline_kw = kw_update(gridline_kw_init, gridline_kw)
    if ticks_corr is None and grid_corr: ticks_corr=ax.get_xticks()
    if ticks_std is None and grid_std: ticks_std=ax.get_yticks()
    if ticks_bias is None and grid_bias: ticks_bias=ax.get_yticks()
    if ticks_corr is not None:
        ax.grid(False, axis='x')
        for theta in ticks_corr:
            if theta == ax.get_thetamin() or theta == ax.get_thetamax(): continue
            ax.plot([theta, theta], [ax.get_rmin(), ax.get_rmax()], color=color_corr, linestyle='--', linewidth=linewidth, alpha=alpha)
    if ticks_std is not None:
        ax.grid(False, axis='y')
        for r in ticks_std:
            if r == ax.get_rmin() or r == ax.get_rmax(): continue
            circle = plt.Circle((0, 0), r, transform=ax.transData._b, 
                                edgecolor=color_std, **gridline_kw)
            ax.add_artist(circle)
    if ticks_bias is not None:
        for r in ticks_bias:
            if r >= 1: continue
            circle = plt.Circle((1, 0), 1-r, transform=ax.transData._b, 
                                edgecolor=color_bias, **gridline_kw)
            ax.add_artist(circle)

# 计算 95 分位数，和 annual total
def cal_P95T(data_array, quantile=0.95, interpolation='midpoint'):
    """
    function_description
    
    Parameters
    ----------
    data_array : xr.DataArray
    
    Returns
    -------
    data_P95 : 95 分位数
    data_P95T : 超过 P95 的年内求和

    
    Notes
    -----
    
    """
    from numpy import nan
    data_P95 = data_array.quantile(quantile, dim='time', interpolation=interpolation, skipna=True)
    data_P95T = xr.where(data_array >= data_P95, data_array, nan).resample(time='Y').sum()
    data_P95.attrs["variable"] = data_array.name
    data_P95.name = 'P95'
    data_P95T.attrs["variable"] = data_array.name
    data_P95T.name = 'P95T'
    return data_P95, data_P95T