# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/10/30 15:12:43 
 
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

def pdf(data, bins):
    return pd.Series(stats.gaussian_kde(data)(bins))

def array_to_frame(array):
    data = array.data.reshape(array.shape[0], -1)
    return pd.DataFrame(data=data, index=array.time).dropna(axis=1)

def pdf_n(array, bins):
    frame = array_to_frame(array)
    return frame.apply(lambda data:pdf(data, bins))

def yearly_sum(array):
    return para.fill_mask(array.resample(time='Y').sum())


def series_plot(ax, data_yr:xr.DataArray, method='diff'):
    year = data_yr.time.dt.year.data

    figu.demean_plot(ax, year, data_yr)
    figu.rolling_mean_plot(ax, year, data_yr)
    if method in 'difference':
        figu.difference_plot(ax, data_yr)
    elif method in 'trend':
        # figu.trend_rollmean_plot(ax, year, data_year)
        figu.trend_plot(ax, year, data_yr, period=slice(1961, 2018))
    else: pass

#%%
# 常量
BINS=np.linspace(0, 35, 140)

# 数据读取
DS = xr.open_dataset(para.prec_path).sel(time=slice('1967', '2018'))

# 数据切片
PREC = DS.prec
EPEG=DS.ep_day
EPE_PREC = PREC.where(EPEG==1)

# 年度统计
TO_PREC_yr = yearly_sum(PREC).mean(dim=['lon', 'lat'])
EPE_PREC_yr = yearly_sum(EPE_PREC).mean(dim=['lon', 'lat'])

TO_FREQ_yr = yearly_sum(np.greater(PREC, 0)).mean(dim=['lon', 'lat'])
EPE_FREQ_yr = yearly_sum(EPEG).mean(dim=['lon', 'lat'])

TO_INTE_yr = TO_PREC_yr / TO_FREQ_yr
EPE_INTE_yr = EPE_PREC_yr / EPE_FREQ_yr

# 频率强度的贡献
Contribute_G = pd.DataFrame({
    'Total Prec': calc.partial_contrib(TO_PREC_yr, 
                                       TO_FREQ_yr, 
                                       TO_INTE_yr), 
    'EPE-G Prec': calc.partial_contrib(EPE_PREC_yr, 
                                       EPE_FREQ_yr, 
                                       EPE_INTE_yr), })

# PDF计算
PREC_P1 = PREC.loc[para.P1]
PREC_P2 = PREC.loc[para.P2]
PREC_P1_PDF = pdf_n(PREC_P1, BINS).mean(axis=1)
PREC_P2_PDF = pdf_n(PREC_P2, BINS).mean(axis=1)
PREC_PDF_DIFF = PREC_P2_PDF - PREC_P1_PDF

#%%
# 可视化
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 2, 3, 4],
                           [5, 6, 7, 8]], figsize=(15, 7), )
axes.format(abcloc='ul', abcborder=True)

axes[0:7].format(xlim=(1965, 2020), 
                 xticks=[1970, 1980, 1990, 2000, 2010, 2020],
                 xticklabels=['1970', '', '1990', '', '2010', ''], 
                 xminorticks=np.arange(1960, 2021, 2.5))

series_plot(axes[0], TO_PREC_yr)
series_plot(axes[1], EPE_PREC_yr)
series_plot(axes[2], TO_FREQ_yr, method='trend')
series_plot(axes[3], EPE_FREQ_yr)
series_plot(axes[4], TO_INTE_yr, method='trend')
series_plot(axes[5], EPE_INTE_yr, method='trend')

figu.bar_plot_from_df(axes[6], Contribute_G, 
                      colors=['blue7', 'red7', 'grey'])

ax8 = axes[7]
line1, = ax8.plot(BINS, PREC_P1_PDF, color='blue7')
line2, = ax8.plot(BINS, PREC_P2_PDF, color='red7')

ax8.vlines([PREC.where(PREC>1).quantile(q=0.33).data, 
            PREC.where(PREC>1).quantile(q=0.67).data], 0, 1, 
            colors='k', ls='--', lw=1.75)

ax8.format(
    xlim=(BINS[0], BINS[-1]), ylim=(0, 0.12), 
    ylabel='Probability Density', xlabel='Precitation (mm/day)'
)

ax82 = ax8.alty(color='grey')

line3, = ax82.plot(BINS, PREC_PDF_DIFF/PREC_P1_PDF, color='grey')
ax82.format(
    ylim=(-0.11, 0.11), 
    yticks=np.arange(-0.10, 0.11, 0.05),
    yticklabels=['-10%', '-5%', '0', '5%', '10%'],
    ylabel='Difference'
)

ax8.legend(
    handles=[line1, line2, line3], 
    labels=['P1', 'P2', 'Diff'], 
    ncols=1, loc='center right')

# fig.savefig('./pics/FIGS4_PDF_PREC.png', dpi=400)
# fig.show()
#%%