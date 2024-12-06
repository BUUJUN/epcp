# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/23 17:02:29 
 
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
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(para)
importlib.reload(calc)

# configuring proplot
pplt.rc.update(
    # name = value, 
    # style = 'seaborn-poster',
    # style = 'default',
    abc = '(a) ',
    # abc = False,
    borders = False,
    grid = False,
)

pplt.rc['figure.dpi'] = 400
pplt.rc['savefig.dpi'] = 800
pplt.rc['figure.facecolor'] = 'w'
# 字
pplt.rc['font.small'] = 16
pplt.rc['font.large'] = 20
pplt.rc['title.loc'] = 'left'

# 子图
pplt.rc['subplots.share'] = False
pplt.rc['subplots.tight'] = True
pplt.rc['subplots.align'] = True
pplt.rc['subplots.innerpad'] = 1
pplt.rc['subplots.outerpad'] = 1
pplt.rc['subplots.panelpad'] = 0
pplt.rc['subplots.panelwidth'] = 1
## 左上角标签
# pplt.rc['abc.loc'] = 'ul'
pplt.rc['abc.loc'] = 'l'
pplt.rc['abc.border'] = False
pplt.rc['abc.bbox'] = False
## 坐标轴
pplt.rc['meta.linewidth'] = 2 # 轴和刻度的厚度
pplt.rc['tick.dir'] = 'out'
pplt.rc['tick.minor'] = True
pplt.rc['tick.len'] = 10
pplt.rc['tick.lenratio'] = 0.4
pplt.rc['tick.widthratio'] = 0.8
pplt.rc['tick.labelpad'] = 1
## 网格
pplt.rc['grid.labels'] = False
pplt.rc['grid.color'] = 'grey'
pplt.rc['gridminor.color'] = 'grey'
pplt.rc['grid.alpha'] = 0.5
pplt.rc['gridminor.alpha'] = 0.5
pplt.rc['grid.linewidth'] = 0.5
pplt.rc['gridminor.linewidth'] = 0.25
## 海岸线
pplt.rc['coast'] = False
pplt.rc['coast.alpha'] = 1
pplt.rc['coast.color'] = 'k'
pplt.rc['coast.zorder'] = 5
###
pplt.rc['lines.linewidth'] = 2.5

# colorbar & legend
pplt.rc['colorbar.loc'] = 'bottom'
pplt.rc['colorbar.length'] = 1
pplt.rc['colorbar.width'] = 0.25
pplt.rc['colorbar.width'] = 0.25
pplt.rc['colorbar.extend'] = 2.5
pplt.rc['colorbar.grid'] = False
pplt.rc['colorbar.rasterize'] = True
pplt.rc['legend.borderaxespad'] = 1
pplt.rc['legend.borderpad'] = 0
pplt.rc['legend.columnspacing'] = 0.75
pplt.rc['legend.handletextpad'] = 0.25
pplt.rc['legend.edgecolor'] = 'None'
pplt.rc['legend.facecolor'] = 'None'
pplt.rc['legend.fancybox' ] = False
pplt.rc['legend.framealpha'] = 0.8
# pplt.rc['legend.fontsize'] = pplt.rc['font.small']

mpl.rc('hatch', color='w', linewidth=1.5)  ##阴影
#%%