# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/06/21 14:24:33 
 
@author: BUUJUN WANG

FIG 2 CNN 架构
"""
#%%
import numpy as np
import xarray as xr
import torch
import proplot as pplt
import matplotlib.pyplot as plt
import buujun.parameters as para
import buujun.figure_1d as figu1
import buujun.figure_2d as figu2
from matplotlib import patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
mpl.rc('font', family='Helvetica')

# #%%
# # 如果报错
# import os
# import importlib
# os.system('rm -rf ~/.cache/matplotlib/fontlist-v*.json')
# importlib.reload(mpl)
# mpl.rc('font', family='Helvetica')
# #%%

## 卷积网络
net_path = para.model_path
net = torch.load(net_path).cpu()
print(net)

slp = xr.open_dataset(para.var_path['msl']).msl.loc[:, para.lat_cnn, para.lon_cnn]
z = xr.open_dataset(para.var_path['z']).z.sel(level=500).loc[:, para.lat_cnn, para.lon_cnn]
u = xr.open_dataset(para.var_path['u']).u.sel(level=850).loc[:, para.lat_cnn, para.lon_cnn]
v = xr.open_dataset(para.var_path['v']).v.sel(level=850).loc[:, para.lat_cnn, para.lon_cnn]
slp_s = (slp.sel(time='2018-08-30T23:00:00') - slp.mean(dim=['time'])) / slp.std(dim=['time'])
z_s = (z.sel(time='2018-08-30T23:00:00') - z.mean(dim=['time'])) / z.std(dim=['time'])
u_s = (u.sel(time='2018-08-30T23:00:00') - u.mean(dim=['time'])) / u.std(dim=['time'])
v_s = (v.sel(time='2018-08-30T23:00:00') - v.mean(dim=['time'])) / v.std(dim=['time'])

lon = slp.longitude.data
lat = slp.latitude.data

#%%
text_kw1 = dict(ha='center', va='top', zorder=10, fontdict=dict(size=30))
text_kw2 = dict(ha='left', va='center', zorder=10, fontweight='bold', fontdict=dict(size=25))
text_kw3 = dict(ha='center', va='top', zorder=10, fontdict=dict(size=25))
text_kw4 = dict(ha='left', va='bottom', zorder=10, fontdict=dict(size=30))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(22.5, 12), dpi=100)
# axes.format(abc=False)
# axes.set_title(r'$\bf{'+'(a)'+'}$ Architecture of CNN', loc='left', 
#                fontsize=fig.get_figwidth()*1.75)
axes.set_title(r'Architecture of CNN', loc='left', 
               fontsize=fig.get_figwidth()*1.75)
axes.set(xlim=(0, 280), ylim=(-75, 75))
axes.axis('off')
# axes.hlines([0], 0, 400)
arrow_len = 8

def add_rectangle(x, y, l, h, ax=axes, lw=1.5, c='red5'):
    rect = patches.Rectangle((x, y), l, h, color=c)
    ax.add_patch(rect)
    rect = patches.Rectangle((x, y), l, h, fill=False, lw=lw)
    ax.add_patch(rect)
    return x+l

def add_cilcle(x, y, r, ax=axes, lw=1.5, c='red5'):
    circ = patches.Circle((x, y), radius=r, color=c)
    ax.add_patch(circ)
    circ = patches.Circle((x, y), radius=r, fill=False, lw=lw)
    ax.add_patch(circ)
    return x+r

def add_arrow(posA, posB, ax=axes, lw=6, c='grey7', 
              arrowstyle='->, head_length=15, head_width=10'):
    arrow = patches.FancyArrowPatch(posA=posA, posB=posB, 
                                    arrowstyle=arrowstyle, 
                                    lw=lw, color=c, mutation_scale=1)
    ax.add_patch(arrow)

x0 = 0
add_rectangle(x0, 26, 32, 32, c='k', lw=2) # 32x32
add_rectangle(x0, -16, 32, 32, c='k', lw=2) # 32x32
x1 = add_rectangle(x0, -26-32, 32, 32, c='k', lw=2)
axes.text(x=(x0+x1)/2, y=73, s='Input\n(standardized)', **text_kw1)
axes.text(x=(x0+x1)/2, y=26-1, s='U850 (31×31)', **text_kw3)
axes.text(x=(x0+x1)/2, y=-16-1, s='V850 (31×31)', **text_kw3)
axes.text(x=(x0+x1)/2, y=-26-32-1, s='Z500 (31×31)', **text_kw3)

x0 = x1+10
add_arrow((x1+1, 0), (x0-1, 0))
add_arrow((x1+1, 26+32/2), (x0-1, 26+32/2))
add_arrow((x1+1, -26-32/2), (x0-1, -26-32/2))

add_rectangle(x0+1, 26+(32-28)/2+1, 28, 28, c='blue5') # 28x28
add_rectangle(x0+1, -16+(32-28)/2+1, 28, 28, c='blue5') # 28x28
add_rectangle(x0+1, -26-32+(32-28)/2+1, 28, 28, c='blue5')

add_rectangle(x0, 26+(32-28)/2, 28, 28, c='blue5') # 28x28
add_rectangle(x0, -16+(32-28)/2, 28, 28, c='blue5') # 28x28
x1 = add_rectangle(x0, -26-32+(32-28)/2, 28, 28, c='blue5')

add_rectangle(x0-1, 26+(32-28)/2-1, 28, 28, c='blue5') # 28x28
add_rectangle(x0-1, -16+(32-28)/2-1, 28, 28, c='blue5') # 28x28
add_rectangle(x0-1, -26-32+(32-28)/2-1, 28, 28, c='blue5')

axes.text(x=(x0+x1)/2, y=73, s='Conv_1\n(4×4)', **text_kw1)
axes.text(x=(x0+x1)/2, y=-16+(32-28)/2-2, s='12×28×28', **text_kw3)
axes.text(x=(x0+x1)/2, y=26+(32-28)/2-2, s='12×28×28', **text_kw3)
axes.text(x=(x0+x1)/2, y=-26-32+(32-28)/2-2, s='12×28×28', **text_kw3)

x0 = x1+11
add_arrow((x1+2, 26+32/2), (x0-1, 26 +32/2))
add_arrow((x1+2, 0), (x0-1, 0))
add_arrow((x1+2, -26-32/2), (x0-1, -26-32/2))

add_rectangle(x0+1, 26+(32-18)/2+1, 18, 18, c='red5') # 14x14
add_rectangle(x0+1, -16+(32-18)/2+1, 18, 18, c='red5') # 14x14
add_rectangle(x0+1, -26-32+(32-18)/2+1, 18, 18, c='red5')

add_rectangle(x0, 26+(32-18)/2, 18, 18, c='red5') # 14x14
add_rectangle(x0, -16+(32-18)/2, 18, 18, c='red5') # 14x14
x1 = add_rectangle(x0, -26-32+(32-18)/2, 18, 18, c='red5')

add_rectangle(x0-1, 26+(32-18)/2-1, 18, 18, c='red5') # 14x14
add_rectangle(x0-1, -16+(32-18)/2-1, 18, 18, c='red5') # 14x14
add_rectangle(x0-1, -26-32+(32-18)/2-1, 18, 18, c='red5')

axes.text(x=(x0+x1)/2, y=73, s='MP\n(2×2)', **text_kw1)
axes.text(x=(x0+x1)/2, y=26+(32-18)/2-2, s='12×14×14', **text_kw3)
axes.text(x=(x0+x1)/2, y=-16+(32-18)/2-2, s='12×14×14', **text_kw3)
axes.text(x=(x0+x1)/2, y=-26-32+(32-18)/2-2, s='12×14×14', **text_kw3)

x0 = x1+13
bracket_style = "]-, widthA=250, lengthA=15"
add_arrow((x1+4, 0), (x0-2, 0), arrowstyle=bracket_style)
add_arrow((x1+4, 0), (x0-1, 0))

add_rectangle(x0+1, -15/2+1, 15, 15, c='blue5') # 11x11
x1 = add_rectangle(x0, -15/2, 15, 15, c='blue5') # 11x11
add_rectangle(x0-1, -15/2-1, 15, 15, c='blue5') # 11x11
axes.text(x=(x0+x1)/2, y=20, s='Conv_2\n(4×4)', **text_kw1)
axes.text(x=(x0+x1)/2, y=-15/2-2, s='48×11×11', **text_kw3)

x0 = x1+11
add_arrow((x1+2, 0), (x0-1, 0))

add_rectangle(x0+1, -10/2+1, 10, 10, c='red5') # 5x5
x1 = add_rectangle(x0, -10/2, 10, 10, c='red5') # 5x5
add_rectangle(x0-1, -10/2-1, 10, 10, c='red5') # 5x5
axes.text(x=(x0+x1)/2, y=20, s='MP\n(2×2)', **text_kw1)
axes.text(x=(x0+x1)/2, y=-10/2-2, s='48×5×5', **text_kw3)

r = 2
x0 = x1+10+r
add_arrow((x1+2, 0), (x0-r, 0))

for y in np.arange(10, 60, 5):
    add_cilcle(x0, y, r, c='red5') # 5x5
    x1 = add_cilcle(x0, -y, r, c='red5') # 5x5
axes.scatter([x0, x0, x0, x0, x0, x0], 
             [1.5, 3.5, 5.5,
              -1.5, -3.5, -5.5,], c='k', s=fig.get_figwidth())
axes.text(x=x0, y=65, s='Flatten', **text_kw1)
axes.text(x=x0, y=-55-r-1, s='1200', **text_kw3)

x0 = x1+30
# add_arrow((x1, 0), (x0-r, 0))
for y1 in np.concatenate([np.arange(10, 60, 5), -np.arange(10, 60, 5)]):
    for y2 in np.concatenate([np.arange(10, 45, 5), -np.arange(10, 45, 5)]):
        axes.plot([x1, x0-r], [y1, y2], color='grey7', lw=0.5)

for y in np.arange(10, 45, 5):
    add_cilcle(x0, y, r, c='green5') # 5x5
    x1 = add_cilcle(x0, -y, r, c='green5') # 5x5
axes.scatter([x0, x0, x0, x0, x0, x0], 
             [1.5, 3.5, 5.5,
              -1.5, -3.5, -5.5,], c='k', s=fig.get_figwidth())
axes.text(x=x0, y=55, s='FC_1\n(dropout)', **text_kw1)
axes.text(x=x0, y=-40-r-1, s='96', **text_kw3)

x0 = x1+30
# add_arrow((x1, 0), (x0-r, 0))
for y1 in np.concatenate([np.arange(10, 45, 5), -np.arange(10, 45, 5)]):
    for y2 in np.concatenate([np.arange(10, 30, 5), -np.arange(10, 30, 5)]):
        axes.plot([x1, x0-r], [y1, y2], color='grey7', lw=0.5)

for y in np.arange(10, 30, 5): 
    add_cilcle(x0, y, r, c='green5') # 5x5
    x1 = add_cilcle(x0, -y, r, c='green5') # 5x5
axes.scatter([x0, x0, x0, x0, x0, x0], 
             [1.5, 3.5, 5.5,
              -1.5, -3.5, -5.5,], c='k', s=fig.get_figwidth())
axes.text(x=x0, y=40, s='FC_2\n(dropout)', **text_kw1)
axes.text(x=x0, y=-25-r-1, s='48', **text_kw3)

x0 = x1+25
r=3  # output
# add_arrow((x1, 0), (x0-3, 0))
for y1 in np.concatenate([np.arange(10, 20, 5), -np.arange(10, 20, 5)]):
    for y2 in np.array([5, -5]):
        axes.plot([x1, x0-r], [y1, y2], color='grey7', lw=0.5)

add_cilcle(x0, 5, r, c='orange5') # 5x5
x1 = add_cilcle(x0, -5, r, c='orange5') # 5x5
axes.text(x=x0, y=18, s='Output', **text_kw1)
axes.text(x=x0+r+1, y=5, s='EPCP', **text_kw2)
axes.text(x=x0+r+1, y=-5, s='NEPCP', **text_kw2)

print(x1)

axes.text(210, -65, 
r'''$\bf{Conv}$: Convolutional layer
$\bf{MP}$: Max-Pooling layer
$\bf{FC}$: Fully Connected layer
$\bf{dropout}$: probability = 0.4''', **text_kw4, 
bbox=dict(facecolor='none', edgecolor='r', pad=10, lw=2))

fig.tight_layout()

px = axes.inset_axes([0, 26, 32, 32], transform=axes.transData)
px.axis('off')
px1 = fig.add_axes([px.get_position().x0, px.get_position().y0, 
                    px.get_position().width, px.get_position().height], 
                   projection=ccrs.PlateCarree())
px1.set_extent([99.5, 130.5, 4.5, 34.5])
px1.add_feature(cfeature.COASTLINE, edgecolor='k', zorder=2)
px1.pcolormesh(lon, lat, u_s, lw=0, cmap='anomaly', vmin=-1.5, vmax=1.5)

px = axes.inset_axes([0, -16, 32, 32], transform=axes.transData)
px.axis('off')
px2 = fig.add_axes([px.get_position().x0, px.get_position().y0, 
                    px.get_position().width, px.get_position().height], 
                   projection=ccrs.PlateCarree())
px2.set_extent([99.5, 130.5, 4.5, 34.5])
px2.add_feature(cfeature.COASTLINE, edgecolor='k', zorder=2)
px2.pcolormesh(lon, lat, v_s, lw=0, cmap='anomaly', vmin=-1.5, vmax=1.5)

px = axes.inset_axes([0, -26-32, 32, 32], transform=axes.transData)
px.axis('off')
px2 = fig.add_axes([px.get_position().x0, px.get_position().y0, 
                    px.get_position().width, px.get_position().height], 
                   projection=ccrs.PlateCarree())
px2.set_extent([99.5, 130.5, 4.5, 34.5])
px2.add_feature(cfeature.COASTLINE, edgecolor='k', zorder=2)
px2.pcolormesh(lon, lat, z_s, lw=0, cmap='anomaly', vmin=-1.5, vmax=1.5)

plt.savefig('./pics/FIG2_CNN架构.png', dpi=400)
plt.show()
# %%
