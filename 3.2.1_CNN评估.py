# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/03/10 21:45:03 
 
@author: BUUJUN WANG
"""
#%%
import xarray as xr
import numpy as np
import pandas as pd
import proplot as pplt
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import metrics
import buujun.parameters as para
import buujun.calculate as calc
import buujun.figure_1d as figu
import buujun.figure_2d as figu2
import importlib
importlib.reload(para)
importlib.reload(calc)
importlib.reload(figu)
importlib.reload(figu2)

# %%
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)
# cnn_res = cnn_res[cnn_res.index.year>=1993]

#%%
metrics.accuracy_score(cnn_res.true_ep, cnn_res.predict_ep)

#%%
fig, axes = pplt.subplots([[1, 2]], figsize=(10, 5))
# axes.format(abc='a', abcloc='l', )

## ***********************
## * 降水量-EPCP概率 散点图
## ***********************
cmap = 'viridis'
nbins = 35
xbins = np.linspace(0, 68, nbins)
ybins = np.linspace(0, 1, nbins)

threshold = cnn_res.precipitation.quantile(1-cnn_res.true_ep.mean())
prec = cnn_res.precipitation
prob = cnn_res.probability

hist, xbins, ybins = np.histogram2d(prec, prob, bins=[xbins, ybins])

sc = axes[0].scatter(prec.values, prob.values, 
                     s=15, color=plt.get_cmap(cmap)([1])[0])
pm = axes[0].pcolormesh(xbins, ybins, 
                        np.where(hist>=6, hist, np.nan).T, 
                        norm='log', cmap=cmap, discrete=False)
lv = axes[0].axvline(threshold, 0, 1, lw=2, color='red', ls='--', label='Threshold')
lh = axes[0].axhline(0.5, 0, 1, lw=2, color='r', ls='-', label='Prob=0.5')

axes[0].colorbar(pm, loc='r', tickminor=True, formatter='log', space=0)
axes[0].legend([lv, lh], loc='lr', ncols=1)
axes[0].format(xlabel='Precipitation (mm/day)', 
               ylabel='Probability', yticks=[0, 0.5, 1], xlim=(0, 65))

to = cnn_res.i.count()
lb = cnn_res.i[(prob<0.5)&(cnn_res.true_ep==0)].count()
lu = cnn_res.i[(prob>=0.5)&(cnn_res.true_ep==0)].count()
rb = cnn_res.i[(prob<0.5)&(cnn_res.true_ep==1)].count()
ru = cnn_res.i[(prob>=0.5)&(cnn_res.true_ep==1)].count()

print(f'''
      Threshold: {threshold:.2f} mm/day
      左下: {lb/to:.2%}
      左上: {lu/to:.2%}
      右下: {rb/to:.2%}
      右上: {ru/to:.2%}
''')

## ***********************
## * 降水量-EPCP占比 直方图
## ***********************
binw = 5
xbins = np.arange(0, 70, 5)

prec_hist, xbins = np.histogram(prec, bins=xbins)
prec_pred0_hist, xbins = np.histogram(prec[cnn_res.predict_ep==0], bins=xbins)
prec_pred1_hist, xbins = np.histogram(prec[cnn_res.predict_ep==1], bins=xbins)
prec_pred0_ratio = prec_pred0_hist/prec_hist
prec_pred1_ratio = prec_pred1_hist/prec_hist
recall = metrics.recall_score(cnn_res.true_ep, cnn_res.predict_ep)
precision = metrics.precision_score(cnn_res.true_ep, cnn_res.predict_ep)

bar_kwards = dict(edgecolor='k', linewidth=0.5)
line_kwards = dict(color = 'k', linewidth=1.5, marker = '.', markersize=10)
bbox_kwards=dict(facecolor='w', edgecolor='None', pad=6, alpha=0.8)
text_kwards=dict(size='xx-large', ha='right', va='bottom', weight='bold', transform=axes[1].transAxes)

ba1 = axes[1].bar(xbins[:-1]+binw/2, prec_pred1_ratio, 
                  color='red3', label='EPCP', **bar_kwards)
ba2 = axes[1].bar(xbins[:-1]+binw/2, prec_pred0_ratio, bottom=prec_pred1_ratio, 
                  color='grey4', label='NEPCP', **bar_kwards)
line= axes[1].plot(xbins[:-1]+binw/2, prec_pred1_ratio, **line_kwards)
lv  = axes[1].axvline(threshold, 0, 1, lw=2, color='red', ls='--', label='Threshold')

axes[1].text(0.95, 0.1, s=f'$recall: {recall:.2f}$\n$precision: {precision:.2f}$', 
             bbox=bbox_kwards, **text_kwards)
axes[1].legend([ba1, ba2], loc='lr', ncols=2, bbox_transform=axes[1].transAxes, 
               bbox_to_anchor=(1.05, 0.95))
axes[1].format(ylabel='Proportion', yticks=[0, 0.5, 1], 
               xlabel='Precipitation (mm/day)', xlim=(0, 65))
axes.format(ylim=(-0.025, 1.025), )

# axes[0].set_title('PL', loc='left')

fig.savefig('./pics/FIG4_CNN评估.png', dpi=400)
fig.show()

#%%
fig, axes = pplt.subplots([[1, 2], [3, 4]], figsize=(10, 10))
axes.format(xloc='bottom', yloc='left')
axes[0:4].format(ylim=(-0.025, 1.025), )
# axes[5].format(ylim=(-0.025, 1.025), )

nepe = xr.open_dataset(para.prec_path).ep_day.sum(dim=['lon', 'lat']).to_series()
cmap = 'viridis'
nbins = 35

P0E1_res = cnn_res[(cnn_res.predict_ep==0)&(cnn_res.true_ep==1)]
P0E1_prec = P0E1_res.precipitation
P0E1_prob = P0E1_res.probability
P0E1_nepe = nepe[(cnn_res.predict_ep==0)&(cnn_res.true_ep==1)]

## ***********************
## * FIG A
## ***********************

xbins = np.linspace(0, 680, nbins)
ybins = np.linspace(0, 1, nbins)
prob = cnn_res.probability
threshold = 205 # 计算得出
hist, xbins, ybins = np.histogram2d(nepe, prob, bins=[xbins, ybins])

sc = axes[0].scatter(nepe.values, prob.values, 
                     s=15, color=plt.get_cmap(cmap)([1])[0])
pm = axes[0].pcolormesh(xbins, ybins, 
                        np.where(hist>=6, hist, np.nan).T, 
                        norm='log', cmap=cmap, discrete=False)
lv = axes[0].axvline(threshold, 0, 1, lw=2, color='k', ls='--', label='Threshold')
lh = axes[0].axhline(0.5, 0, 1, lw=2, color='k', ls='-', label='Prob=0.5')

axes[0].colorbar(pm, loc='r', tickminor=True, formatter='log', space=0)
axes[0].legend([lv, lh], loc='lr', ncols=1)
axes[0].format(xlabel='Number of Grids Exceeding P90', 
           ylabel='Probability', 
           yticks=[0, 0.5, 1], ylim=(-0.05, 1.05))
axes[0].scatter(P0E1_nepe.values, P0E1_prob.values,
                s=3, c='w', zorder=20)

binw = 5
xbins = np.linspace(0, 600, 13)

nepe_hist, xbins = np.histogram(nepe, bins=xbins)
nepe_pred0_hist, xbins = np.histogram(nepe[cnn_res.predict_ep==0], bins=xbins)
nepe_pred1_hist, xbins = np.histogram(nepe[cnn_res.predict_ep==1], bins=xbins)
nepe_pred0_ratio = nepe_pred0_hist/nepe_hist
nepe_pred1_ratio = nepe_pred1_hist/nepe_hist
recall = metrics.recall_score(cnn_res.true_ep, cnn_res.predict_ep)
precision = metrics.precision_score(cnn_res.true_ep, cnn_res.predict_ep)

bar_kwards = dict(edgecolor='k', linewidth=0.5)
line_kwards = dict(color = 'k', linewidth=3, marker = '.', markersize=10)
bbox_kwards=dict(facecolor='w', edgecolor='None', pad=6, alpha=0.8)
text_kwards=dict(size='xx-large', ha='right', va='bottom', weight='bold', transform=axes[1].transAxes)

line= axes[1].plot(xbins[:-1]+binw/2, nepe_pred1_ratio, **line_kwards)
lv  = axes[1].axvline(threshold, 0, 1, lw=2, color='k', ls='--', label='Threshold')

axes[1].text(0.95, 0.1, s=f'$recall: {recall:.2f}$\n$precision: {precision:.2f}$', 
             bbox=bbox_kwards, **text_kwards)
axes[1].format(ylabel='Proportion', yticks=[0, 0.5, 1], ylim=(-0.05, 1.05), 
               xticks=np.arange(0, 601, 200), xlim=(-30, 630),
               xlabel='Number of Grids Exceeding P90')


## ***********************
## * FIG B
## ***********************
xbins = np.linspace(0, 68, nbins)
ybins = np.linspace(0, 1, nbins)
threshold = cnn_res.precipitation.quantile(1-cnn_res.true_ep.mean())
prec = cnn_res.precipitation

hist, xbins, ybins = np.histogram2d(prec, prob, bins=[xbins, ybins])

sc = axes[2].scatter(prec.values, prob.values, 
                     s=15, color=plt.get_cmap(cmap)([1])[0])
pm = axes[2].pcolormesh(xbins, ybins, 
                    np.where(hist>=6, hist, np.nan).T, 
                    norm='log', cmap=cmap, discrete=False)
lv = axes[2].axvline(threshold, 0, 1, lw=2, color='k', ls='--', label='Threshold')
lh = axes[2].axhline(0.5, 0, 1, lw=2, color='k', ls='-', label='Prob=0.5')
axes[2].colorbar(pm, loc='r', tickminor=True, formatter='log', space=0)
axes[2].legend([lv, lh], loc='lr', ncols=1)
axes[2].format(xlabel='Precipitation (mm/day)', 
               ylabel='Probability', yticks=[0, 0.5, 1], xlim=(0, 65))
axes[2].scatter(P0E1_prec.values, P0E1_prob.values,
                s=3, c='w', zorder=20)


binw = 5
xbins = np.arange(0, 70, 5)

prec_hist, xbins = np.histogram(prec, bins=xbins)
prec_pred0_hist, xbins = np.histogram(prec[cnn_res.predict_ep==0], bins=xbins)
prec_pred1_hist, xbins = np.histogram(prec[cnn_res.predict_ep==1], bins=xbins)
prec_pred0_ratio = prec_pred0_hist/prec_hist
prec_pred1_ratio = prec_pred1_hist/prec_hist
recall = metrics.recall_score(cnn_res.true_ep, cnn_res.predict_ep)
precision = metrics.precision_score(cnn_res.true_ep, cnn_res.predict_ep)

line= axes[3].plot(xbins[:-1]+binw/2, prec_pred1_ratio, **line_kwards)
lv  = axes[3].axvline(threshold, 0, 1, lw=2, color='k', ls='--', label='Threshold')

axes[3].format(ylabel='Proportion', yticks=[0, 0.5, 1], 
               xlabel='Precipitation (mm/day)', xlim=(0, 65))

## ***********************
## * FIG C
## ***********************

# xbins = np.linspace(0, 68, nbins)
# ybins = np.linspace(0, 680, nbins)
# threshold = cnn_res.precipitation.quantile(1-cnn_res.true_ep.mean())
# hist, xbins, ybins = np.histogram2d(prec, nepe, bins=[xbins, ybins])

# sc = axes[4].scatter(prec.values, nepe.values, 
#                      s=15, color=plt.get_cmap(cmap)([1])[0])

# pm = axes[4].pcolormesh(xbins, ybins, 
#                         np.where(hist>=20, hist, np.nan).T, 
#                         norm='log', cmap=cmap, discrete=False)

# axes[4].colorbar(pm, loc='r', tickminor=True, formatter='log', space=0)

# lv = axes[4].axvline(threshold, 0, 1, lw=2, color='k', ls='--', label='Threshold')
# lh = axes[4].axhline(205, 0, 1, lw=2, color='k', ls='-', label='Threshold')

# axes[4].legend([lv, lh], loc='lr', ncols=1)
# axes[4].format(xlabel='Precipitation (mm/day)', 
#                ylabel='Number of Grids Exceeding P90', 
#                yticks=np.arange(0, 601, 200),
#                ylim=(-34, 714), xlim=(-3.4, 71.4))


# axes[4].scatter(P0E1_prec.values, P0E1_nepe.values,
#                 s=10, c='w', zorder=20)

# binw = 5
# xbins = np.arange(0, 70, 5)

# prec_hist, xbins = np.histogram(prec, bins=xbins)
# prec_eper0_hist, xbins = np.histogram(prec[cnn_res.true_ep==0], bins=xbins)
# prec_eper1_hist, xbins = np.histogram(prec[cnn_res.true_ep==1], bins=xbins)
# prec_eper0_ratio = prec_eper0_hist/prec_hist
# prec_eper1_ratio = prec_eper1_hist/prec_hist
# recall = metrics.recall_score(cnn_res.true_ep, cnn_res.predict_ep)
# precision = metrics.precision_score(cnn_res.true_ep, cnn_res.predict_ep)

# bar_kwards = dict(edgecolor='k', linewidth=0.5)
# line_kwards = dict(color = 'k', linewidth=3, marker = '.', markersize=10)
# bbox_kwards=dict(facecolor='w', edgecolor='None', pad=6, alpha=0.8)
# text_kwards=dict(size='xx-large', ha='right', va='bottom', weight='bold', transform=axes[5].transAxes)

# # ba1 = axes[5].bar(xbins[:-1]+binw/2, prec_eper1_ratio, 
# #                   color='red3', label='EPCP', **bar_kwards)
# # ba2 = axes[5].bar(xbins[:-1]+binw/2, prec_eper0_ratio, bottom=prec_eper1_ratio, 
# #                   color='grey4', label='NEPCP', **bar_kwards)
# # axes[5].legend([ba1, ba2], loc='lr', ncols=2, bbox_transform=axes[5].transAxes, 
# #                bbox_to_anchor=(1.05, 0.95))
# line= axes[5].plot(xbins[:-1]+binw/2, prec_eper1_ratio, **line_kwards)
# lv  = axes[5].axvline(threshold, 0, 1, lw=2, color='k', ls='--', label='Threshold')

# # axes[5].text(0.95, 0.1, s=f'$recall: {recall:.2f}$\n$precision: {precision:.2f}$', 
# #              bbox=bbox_kwards, **text_kwards)
# axes[5].format(ylabel='Proportion', yticks=[0, 0.5, 1], ylim=(-0.05, 1.05), 
#                xticks=np.arange(0, 70, 20), xlim=(-3.5, 73.5),
#                xlabel='Precipitation (mm/day)')

fig.savefig('./pics/FIG5_CNN评估_3.png', dpi=400)
fig.show()

#%%
fig, axes = pplt.subplots([[1]], figsize=(7.5, 5))
# axes.format(abc='a', abcloc='l', )

## ***********************
## * 降水量-EPCP概率 散点图
## ***********************
nbins = 35
xbins = np.linspace(0, 68, nbins)
ybins = np.linspace(0, 1, nbins)
threshold = cnn_res.precipitation.quantile(1-cnn_res.true_ep.mean())

epe = cnn_res.true_ep == 1

prec_epe = cnn_res.precipitation[epe]
prob_epe = cnn_res.probability[epe]
prec_nepe = cnn_res.precipitation[~epe]
prob_nepe = cnn_res.probability[~epe]

hist_epe, _, _ = np.histogram2d(prec_epe, prob_epe, bins=[xbins, ybins])
hist_nepe, _, _ = np.histogram2d(prec_nepe, prob_nepe, bins=[xbins, ybins])

s1 = axes.scatter([-1], [-1], c='grey4', label=r'NEPE-R')
s2 = axes.scatter([-1], [-1], c='red', label=r'EPE-R')

axes.scatter(cnn_res.precipitation.values, 
                  cnn_res.probability.values, 
                  c=np.where(cnn_res.true_ep.values==0, 1, np.nan), 
                  cmap=pplt.Colormap(['grey4']), 
                  s=10)

axes.scatter(cnn_res.precipitation.values, 
                  cnn_res.probability.values, 
                  c=np.where(cnn_res.true_ep.values==1, 1, np.nan), 
                  cmap=pplt.Colormap(['red']), 
                  s=10)

lv = axes.axvline(threshold, 0, 1, lw=2, color='k', ls='--', label='Threshold')
lh = axes.axhline(0.5, 0, 1, lw=2, color='k', ls='-', label='Prob=0.5')

# axes.colorbar(pm, loc='r', tickminor=True, formatter='log', space=0)
axes.legend([s1, s2, lv, lh], loc='lr', ncols=1)
axes.format(xlabel='Precipitation (mm/day)', 
            ylabel='Probability', yticks=[0, 0.5, 1], xlim=(0, 65),
            ylim=(-0.025, 1.025), 
            xloc='bottom', yloc='left')

fig.savefig('./pics/FIGS4_CNN评估_EPE阈值.png')
fig.show()
# %%
