# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/02/27 10:44:46 
 
@author: BUUJUN WANG

FIG 3 基于格点计算的
(a) 总降水
(b) 极端降水
(c) 频率强度相对贡献
(d) 降水频率
(e) 极端降水频率
(f) 平均雨强
(g) 极端降水强度

FIG S3 基于区域计算的 (同上)

FIG S4 降水量的 PDF
"""
#%%
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, YearBegin
import xarray as xr
import proplot as pplt
from scipy import stats

import importlib
import buujun.figure_1d as figu
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(para)
importlib.reload(calc)

#%%
ds = xr.open_dataset(para.prec_path).sel(time=slice('1961', '2018'))
demean = lambda data_array:data_array-data_array.mean(dim='time')

#%% 数据处理
to_prec_yr = para.fill_mask(
    ds.prec.where(np.greater(ds.prec, 1)).resample(time='Y').sum()).mean(dim=['lon', 'lat'])
# ep_prec = xr.where((ds.ep_day==1)|(np.isnan(ds.ep_day)), ds.prec, np.nan)
ep_prec = ds.prec.where(ds.ep_day==1)
ep_prec_yr = para.fill_mask(ep_prec.resample(time='Y').sum()).mean(dim=['lon', 'lat'])
to_freq_yr = para.fill_mask(np.greater(ds.prec, 1).resample(time='Y').sum()).mean(dim=['lon', 'lat'])
ep_freq_yr = para.fill_mask(ds.ep_day.resample(time='Y').sum()).mean(dim=['lon', 'lat'])
to_inte_yr = to_prec_yr / to_freq_yr
ep_inte_yr = ep_prec_yr / ep_freq_yr

prec_avg = ds.prec.where(np.greater(ds.prec, 1)).mean(dim=['lon', 'lat'])
ep_day_sc = ds.ep_day_sc
ep_prec_sc = xr.where(ds.ep_day_sc==1, prec_avg, np.nan)
to_prec_yr_sc = to_prec_yr
ep_prec_yr_sc = ep_prec_sc.resample(time='Y').sum()
to_freq_yr_sc = to_freq_yr
ep_freq_yr_sc = ep_day_sc.resample(time='Y').sum()
to_inte_yr_sc = to_prec_yr_sc / to_freq_yr_sc
ep_inte_yr_sc = ep_prec_yr_sc / ep_freq_yr_sc

year = to_prec_yr.time.dt.year

contrib_df = pd.DataFrame({
    'Wet Prec': calc.partial_contrib(to_prec_yr, 
                                          to_freq_yr, 
                                          to_inte_yr), 
    'P90 Prec': calc.partial_contrib(ep_prec_yr, 
                                         ep_freq_yr, 
                                         ep_inte_yr), 
})

contrib_df_sc = pd.DataFrame({
    'Wet Prec': calc.partial_contrib(to_prec_yr_sc, 
                                       to_freq_yr_sc, 
                                          to_inte_yr_sc), 
    'REPE Prec': calc.partial_contrib(ep_prec_yr_sc, 
                                         ep_freq_yr_sc, 
                                         ep_inte_yr_sc), 
})

def series_plot(axes, data_year:xr.DataArray, ylabel_1, method='diff'):
    year = data_year.time.dt.year.data
    axes[0].format(xlim=(1960, 2020), xticks=[1960, 1970, 1980, 1990, 2000, 2010, 2020],
                   xticklabels=['', '1970', '', '1990', '', '2010', ''], 
                   xminorticks=np.arange(1960, 2021, 2.5))

    figu.demean_plot(axes[0], year, data_year.data)
    figu.rolling_mean_plot(axes[0], year, data_year.data)
    if 'diff' in method:
        figu.difference_plot(axes[0], data_year, c='k', ls='-', lw=1.5)
    if 'trend' in method:
        # figu.trend_rollmean_plot(axes[0], year, data_year)
        year=year[9:-3]; data_year=data_year[9:-3]
        figu.trend_plot(axes[0], year, data_year, period=slice(1961, 2018), 
                        c='#3BFF3D', ls='--', lw=1.5)

    axes[0].format(ylabel='')
    axes[0].set_title(ylabel_1, loc='left')


def rolling_ttest_plot(axes, data_year, ylabel_1, detrend=False):
    year = data_year.time.dt.year.data
    axes[0].format(xlim=(1960, 2020), xticks=[1960, 1970, 1980, 1990, 2000, 2010, 2020],
                   xticklabels=['', '1970', '', '1990', '', '2010', ''], 
                   xminorticks=np.arange(1960, 2021, 2.5))
    if detrend:
        data_year = calc.detrend(data_year)
    ax1 = axes[0]
    figu.rolling_ttest_plot(ax1, year, data_year, alpha=0.05)
    axes[0].format(ylabel='')
    axes[0].set_title(ylabel_1, loc='left')


def MK_plot(axes, data_year, ylabel_1, detrend=False):
    year = data_year.time.dt.year.data
    axes[0].format(xlim=(1960, 2020), xticks=[1960, 1970, 1980, 1990, 2000, 2010, 2020],
                   xticklabels=['', '1970', '', '1990', '', '2010', ''], 
                   xminorticks=np.arange(1960, 2021, 2.5))
    if detrend:
        data_year = calc.detrend(data_year)
    ax1 = axes[0]
    figu.mktest_plot(ax1, year, data_year, alpha=0.05)
    axes[0].format(ylabel='')
    axes[0].set_title(ylabel_1, loc='left')

#%%
## ***********************
## * 降水序列（格点定义）
## ***********************
importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1,1,1, 2,2,2, 3,3,3, 4,4,4],
     [5,5,5, 6,6,6, 0, 7,7,7,7, 0]], figsize=(13, 6))
series_plot(axes[0:1], to_prec_yr, 'Wet Prec')
series_plot(axes[1:2], ep_prec_yr, 'P90 Prec')
series_plot(axes[2:3], to_freq_yr, 'Wet Freq')
series_plot(axes[3:4], ep_freq_yr, 'P90 Freq')
series_plot(axes[4:5], to_inte_yr, 'Wet Intensity')
series_plot(axes[5:6], ep_inte_yr, 'P90 Intensity', method=['diff', 'trend'])
figu.bar_plot_from_df(axes[6], contrib_df, colors=['blue7', 'red7', 'grey'])

axes[6].format(yticklabels=[f'{i:.0%}' for i in axes[6].get_yticks()])
axes[6].format(ylabel='', ylim=(-0.05, 0.8))
axes[6].set_title('Contributions', loc='left')

import matplotlib.pyplot as plt
# plt.savefig('./pics/FIG3_OBS_PREC.png', dpi=400)
plt.show()

#%%
## ***********************
## * 滑动T检验
## ***********************
importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1,1,1, 2,2,2, 3,3,3, 4,4,4],
     [5,5,5, 6,6,6, 0, 0,0,0,0, 0]], figsize=(13, 5))
axes.format(xloc='bottom', yloc='left')

rolling_ttest_plot(axes[0:1], to_prec_yr, 'Wet Prec')
rolling_ttest_plot(axes[1:2], ep_prec_yr, 'P90 Prec')
rolling_ttest_plot(axes[2:3], to_freq_yr, 'Wet Freq')
rolling_ttest_plot(axes[3:4], ep_freq_yr, 'P90 Freq')
rolling_ttest_plot(axes[4:5], to_inte_yr, 'Wet Intensity')
rolling_ttest_plot(axes[5:6], ep_inte_yr, 'P90 Intensity')

fig.format(suptitle='Moving t-test', ylim=(-0.05, 1))
# fig.savefig('./pics/FIG3_MT-test.png')
fig.show()

#%%
## ***********************
## * MK
## ***********************
importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1,1,1, 2,2,2, 3,3,3, 4,4,4],
     [5,5,5, 6,6,6, 0, 0,0,0,0, 0]], figsize=(13, 7))
axes.format(xloc='bottom', yloc='left')

MK_plot(axes[0:1], to_prec_yr, 'Wet Prec')
MK_plot(axes[1:2], ep_prec_yr, 'P90 Prec')
MK_plot(axes[2:3], to_freq_yr, 'Wet Freq')
MK_plot(axes[3:4], ep_freq_yr, 'P90 Freq')
MK_plot(axes[4:5], to_inte_yr, 'Wet Intensity')
MK_plot(axes[5:6], ep_inte_yr, 'P90 Intensity')

fig.format(suptitle='MK test')
# fig.savefig('./pics/FIG3_MT-test.png')
fig.show()


#%%
## ***********************
## * 降水序列（区域）
## ***********************
importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1,1,1, 2,2,2, 3,3,3, 4,4,4],
     [5,5,5, 6,6,6, 0, 7,7,7,7, 0]], figsize=(13, 6))
series_plot(axes[0:1], to_prec_yr_sc, 'Wet Prec')
series_plot(axes[1:2], ep_prec_yr_sc, 'REPE Prec')
series_plot(axes[2:3], to_freq_yr_sc, 'Wet Freq')
series_plot(axes[3:4], ep_freq_yr_sc, 'REPE Freq')
series_plot(axes[4:5], to_inte_yr_sc, 'Wet Intensity')
series_plot(axes[5:6], ep_inte_yr_sc, 'REPE Intensity', method=['diff', 'trend'])
figu.bar_plot_from_df(axes[6], contrib_df_sc, 
                      colors=['blue7', 'red7', 'grey'])

axes[6].format(yticklabels=[f'{i:.0%}' for i in axes[6].get_yticks()])
axes[6].format(ylabel='', ylim=(-0.05, 0.8))
axes[6].set_title('Contributions', loc='left')

# plt.savefig('./pics/FIGS3_OBS_PREC.png', dpi=400)
plt.show()

#%%
## ***********************
## * 分月贡献
## ***********************
def month_plot(axes, data_mon:xr.DataArray):
    y = figu.preprocessing_1d(data_mon)
    res_mon = figu.to_MultiIndex_frame(y).apply(lambda series:figu.series_diff(series, pe=para.P1, pl=para.P2))
    diff_mon = res_mon.loc['difference']
    x = diff_mon.index.values

    axes.format(xlim=(x[0]-1, x[-1]+1), xticks=x, xminorticks=[])
    rects = axes.bar(x, diff_mon.values, color='violet5', ec='k', lw=1, **figu.kwards_bar)
    axes.plot([axes.get_xlim()[0], axes.get_xlim()[1]], [0, 0], **figu.kwards_zero_line)

    axes.bar_label(rects, 
                   labels=np.where(res_mon.loc['pvalue'].values<=0.1, '*', ''),
                   padding=-40, color='white', fontweight='bold', fontsize=30)
    print('Values of difference: ', diff_mon.values)
    print('Values of difference (Wet): ', diff_mon.values.sum())
    print('T-test for monthly difference: \n', res_mon.loc['pvalue'].values, '\n')

to_prec_mon = ds.prec.mean(dim=['lon', 'lat'])\
    .to_series().groupby(MonthBegin().rollback).sum()
ep_prec_mon = para.fill_mask(ep_prec.resample(time='M').sum()).mean(dim=['lon', 'lat']).to_series().dropna()
ep_freq_mon = ds.ep_day.mean(dim=['lon', 'lat'])\
    .to_series().groupby(MonthBegin().rollback).sum()

importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1, 2, 3]], figsize=(12, 4), dpi=100)
month_plot(axes[0:1], to_prec_mon, )
month_plot(axes[1:2], ep_prec_mon, )
month_plot(axes[2:3], ep_freq_mon, )
axes[0].format(title='Wet Prec', ylabel='')
axes[1].format(title='P90 Prec', ylabel='')
axes[2].format(title='P90 Freq', ylabel='')

# fig.savefig('./pics/FIGS7_Change_Months.png', dpi=400)
fig.show()

rep_prec = xr.where(ds.ep_day_sc==1, ds.prec, np.nan)
rep_prec_mon = rep_prec.mean(dim=['lon', 'lat']).to_series().groupby(MonthBegin().rollback).sum()
rep_freq_mon = ds.ep_day_sc.to_series().groupby(MonthBegin().rollback).sum()
rep_inte_mon = rep_prec_mon / rep_freq_mon

importlib.reload(figu)
fig, axes = pplt.subplots(
    [[1, 2, 3]], figsize=(12, 4), dpi=100)
month_plot(axes[0:1], to_prec_mon, )
month_plot(axes[1:2], rep_prec_mon, )
month_plot(axes[2:3], rep_freq_mon, )
axes[0].format(title='Wet Prec', ylabel='')
axes[1].format(title='REPE Prec', ylabel='')
axes[2].format(title='REPE Freq', ylabel='')

# fig.savefig('./pics/FIGS7_Change_Months_REPE.png', dpi=400)
fig.show()

# def contribute_plot(axes, data_mon:xr.DataArray, ):
#     figu.diff_contribute(axes[0], data_mon)
#     axes[0].format(ylabel='Contributions', xticks=np.arange(4, 9),
#                    xticklabels=['Apr', 'May', 'Jun', 'Jul', 'Aug'],
#                    yticks=[0, 0.2, 0.4, 0.6], yticklabels=['0%', '20%', '40%', '60%'])

# importlib.reload(figu)
# fig, axes = pplt.subplots(
#     [[1, 2, 3]], figsize=(12, 4), dpi=100)
# contribute_plot(axes[0:1], to_prec_mon, )
# contribute_plot(axes[1:2], ep_prec_mon, )
# contribute_plot(axes[2:3], ep_freq_mon, )
# axes[0].format(title='Total Prec')
# axes[1].format(title='GEPE Prec')
# axes[2].format(title='GEPE Freq')

# fig.savefig('./pics/FIGS7_Contrib_Months.png', dpi=400)
# fig.show()

#%%
## ***********************
## * Pdf
## ***********************
def pdf(data, bins):
    return pd.Series(stats.gaussian_kde(data)(bins))

def dataarray_dataframe(data_array):
    data = data_array.data.reshape(data_array.shape[0], -1)
    return pd.DataFrame(data=data, index=data_array.time).dropna(axis=1)

def pdf_n(data_array, bins):
    data_frame = dataarray_dataframe(data_array)
    return data_frame.apply(lambda data:pdf(data, bins))

bins = np.linspace(0, 35, 140)
prec_pe_pdf = pdf_n(ds.prec.loc[para.PE], bins).mean(axis=1)
prec_pl_pdf = pdf_n(ds.prec.loc[para.PL], bins).mean(axis=1)

importlib.reload(figu)

with pplt.rc.context(abc=False):
    fig, axes = pplt.subplots([[1]], figsize=(7, 5), abc=False)


    ax1 = axes
    line1, = ax1.plot(bins, prec_pe_pdf, color='blue7')
    line2, = ax1.plot(bins, prec_pl_pdf, color='red7')

    ax1.vlines([ds.prec.where(ds.prec>1).quantile(q=0.33).data, 
                ds.prec.where(ds.prec>1).quantile(q=0.67).data], 0, 1, 
                colors='k', ls='--', lw=1.75, zorder=5)

    ax1.format(
        xlim=(bins.min(), bins.max()), ylim=(0, 0.12),
        ylabel='Probability Density', xlabel='Precitation'
    )

    ax2 = axes.alty(color='grey')
    line3, = ax2.plot(bins, (prec_pl_pdf-prec_pe_pdf)/prec_pe_pdf, color='grey')
    ax2.format(
        ylim=(-0.11, 0.11), 
        yticks=np.arange(-0.10, 0.11, 0.05),
        yticklabels=['-10%', '-5%', '0', '5%', '10%'],
        ylabel='Change'
    )

    axes.legend(
        handles=[line1, line2, line3], 
        labels=['P1', 'P2', 'Change'], 
        ncols=1, loc='center right')

    # fig.savefig('./pics/FIGS4_PDF_PREC.png', dpi=400)
    fig.show()

# %%
