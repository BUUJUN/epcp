# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/03/07 15:51:37 
 
@author: BUUJUN WANG
"""

import numpy as np
import proplot as pplt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def add_axis_axes(ax_base, **kwargs):
    locator = ax_base._make_inset_locator([0, 0, 1, 1], ax_base.transAxes)
    ax_add = pplt.CartesianAxes(ax_base.figure, locator(ax_base, None).bounds, **kwargs)
    ax_add.set_axes_locator(locator)
    ax_axis = ax_base.add_child_axes(ax_add)
    ax_axis.format(grid=False, xlim=ax_base.get_xlim(), ylim=ax_base.get_ylim())
    return ax_axis


def find_side(outline_patch, tick_position):
    minx, miny, maxx, maxy = outline_patch.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[tick_position])


def _lambert_ticks(ax, ticks, tick_position, tick_extractor, line_constructor):
    outline_patch = sgeom.LineString(ax.spines['geo'].get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_position)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs: tick = [None]
        else: tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = np.asarray(ticks).tolist()
    while True:
        try: index = _ticks.index(None)
        except ValueError: break
        _ticks.pop(index)
        ticklabels.pop(index)
        
    return _ticks, ticklabels


def lambert_ticks(ax, ticks, tick_position='bottom', which='major', **kwards_add_axes):

    if tick_position in ['top', 'right']: 
        ax_axis = add_axis_axes(ax_base=ax, **kwards_add_axes)
    else: ax_axis = ax

    if tick_position in ['top', 'bottom']:
        tick_extractor = lambda xy: xy[0]
        line_constructor = lambda t, n_steps, extent: np.vstack((
            np.zeros(n_steps)+t, 
            np.linspace(extent[2]+0.1*(extent[2]-extent[3]), 
                        extent[3]-0.1*(extent[2]-extent[3]), n_steps))).T
        ticks, ticklabels = _lambert_ticks(ax, ticks, tick_position, 
                                           tick_extractor, line_constructor)
        if which == 'major':
            ax_axis.set_xticks(ticks)
            ax_axis.set_xticklabels([LongitudeFormatter()(tick) for tick in ticklabels])
        elif which == 'minor':
            ax_axis.xaxis.set_minor_locator(mticker.FixedLocator(ticks))
        else: ValueError("Input which must in ['major', 'minor'].")

    elif tick_position in ['left', 'right']:
        tick_extractor = lambda xy: xy[1]
        line_constructor = lambda t, n_steps, extent: np.vstack((
            np.linspace(extent[0], extent[1], n_steps), np.zeros(n_steps)+t)).T
        ticks, ticklabels = _lambert_ticks(ax, ticks, tick_position, 
                                           tick_extractor, line_constructor)
        if which == 'major':
            ax_axis.set_yticks(ticks)
            ax_axis.set_yticklabels([LatitudeFormatter()(tick) for tick in ticklabels])
        elif which == 'minor':
            ax_axis.yaxis.set_minor_locator(mticker.FixedLocator(ticks))
        else: ValueError("Input which must in ['major', 'minor'].")
        
    else: 
        raise ValueError("Input tick_position must in ['top', 'bottom', 'left', 'right'].")
    
    return ax_axis

