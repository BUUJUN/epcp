# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/10/25 06:19:05 
 
@author: BUUJUN WANG
"""
#%%
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import proplot as pplt
import cartopy.crs as ccrs
from scipy import stats
from scipy import signal
import buujun.figure_1d as figu
import buujun.figure_2d as figu2
import buujun.parameters as para
import buujun.calculate as calc
importlib.reload(figu)
importlib.reload(figu2)
importlib.reload(para)
importlib.reload(calc)

#%%
AM = [4, 5]; days_AM=30+31
JJA = [6, 7, 8]; days_JJA=30+31+31

columns = ['ALL', 'NEPCP', 'EPCP']
labels = ['Total', 'Dyn', 'Thermo', 'Inter']
labels_long = ['Total', 'Dynamic', 'Thermodynamic', 'Interaction']
colors = ['orange', 'blue7', 'red7', 'grey']

CNN_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)\
                        .loc['1967-01-01':'2018-12-31']

PREC_ds = xr.open_dataset(para.prec_path).sel(time=slice('1967', '2018'))

year = np.asarray(list(set(CNN_res.index.year.values)), dtype='int')

lon = PREC_ds.lon.data
lat = PREC_ds.lat.data

def partition(patterns, var):
    f_mean = np.empty((2, year.size))
    f_dist = np.empty((2, year.size))
    v_mean = np.empty((2, year.size))
    v_dist = np.empty((2, year.size))

    for i in [0, 1]:
        freq_yr = (patterns==i).resample('Y').sum()

        var_pi_inte_yr = var.loc[patterns.index[patterns==i]]\
            .resample(time='Y').sum() / xr.DataArray(freq_yr)
        
        f_mean[i] = freq_yr.mean()
        f_dist[i] = freq_yr - f_mean[i]
        v_mean[i] = xr.where(np.isnan(var_pi_inte_yr), np.nan, 
                             var_pi_inte_yr.mean(dim='time'))
        v_dist[i] = var_pi_inte_yr - v_mean[i]
    
    f_dist = np.where(np.isnan(f_dist), 0, f_dist)

    dims = ['pattern', 'time']
    coords = dict(pattern=['NEPCP', 'EPCP'], time=year)
    v_mean_array = xr.DataArray(data=v_mean, dims=dims, coords=coords)
    v_dist_array = v_mean_array.copy(data=v_dist)
    f_mean_array = v_mean_array.copy(data=f_mean)
    f_dist_array = v_mean_array.copy(data=f_dist)

    therm = f_mean_array*v_dist_array
    dynam = f_dist_array*v_mean_array
    nolin = f_dist_array*v_dist_array # - (f_dist*p_dist).mean(axis=0)
    total = therm + dynam + nolin

    res_dt = pd.DataFrame(
        data={'total':  total.sum(dim='pattern'), 
              'dynam':  dynam.sum(dim='pattern'), 
              'therm':  therm.sum(dim='pattern'), 
              'nolin':  nolin.sum(dim='pattern'), 
              'total_0':total[0], 
              'dynam_0':dynam[0], 
              'therm_0':therm[0], 
              'nolin_0':nolin[0], 
              'total_1':total[1], 
              'dynam_1':dynam[1], 
              'therm_1':therm[1], 
              'nolin_1':nolin[1], }, 
        index=pd.date_range('1967-01-01', '2018-12-31', freq='y'))
    
    return res_dt

def statics(series):
    slope=stats.linregress(x=year, y=series).slope
    pvalue=stats.linregress(x=year, y=series).pvalue
    late=series.loc[para.PL]
    early=series.loc[para.PE]
    delta=late.mean()-early.mean()
    ttest=stats.ttest_ind(a=late, b=early).pvalue
    return pd.Series(dict(slope=slope, pvalue=pvalue, delta=delta, ttest=ttest))

#%%
CNN_res_AM = CNN_res[np.isin(CNN_res.index.month, AM)]
Patterns_AM = CNN_res_AM.predict_ep
PREC_ds_AM = PREC_ds.sel(time=PREC_ds.time[np.isin(PREC_ds.time.dt.month, AM)])
P_REPE_AM = PREC_ds_AM.prec.mean(dim=['lon', 'lat'])
RES_AM = partition(Patterns_AM, P_REPE_AM)
RES_statics_AM = RES_AM.apply(statics)
Change_AM = RES_statics_AM.loc['delta']
Change_AM_dt = pd.DataFrame(
    np.asarray([Change_AM[:4], Change_AM[4:8], Change_AM[8:]]).T,
    index=labels, columns=columns)
Contrib_AM = Change_AM_dt.iloc[1:] / Change_AM_dt.iloc[0]

CNN_res_JJA = CNN_res[np.isin(CNN_res.index.month, JJA)]
Patterns_JJA = CNN_res_JJA.predict_ep
PREC_ds_JJA = PREC_ds.sel(time=PREC_ds.time[np.isin(PREC_ds.time.dt.month, JJA)])
P_REPE_JJA = PREC_ds_JJA.prec.mean(dim=['lon', 'lat'])
RES_JJA = partition(Patterns_JJA, P_REPE_JJA)
RES_statics_JJA = RES_JJA.apply(statics)
Change_JJA = RES_statics_JJA.loc['delta']
Change_JJA_dt = pd.DataFrame(
    np.asarray([Change_JJA[:4], Change_JJA[4:8], Change_JJA[8:]]).T,
    index=labels, columns=columns)
Contrib_JJA = Change_JJA_dt.iloc[1:] / Change_JJA_dt.iloc[0]

#%%
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 2]], figsize=(10, 6))
axes.format(xloc='bottom', yloc='left')

kwards_bar = dict(colors=colors, width=1, ec='k', lw=1, legend=False)

ax1 = axes[0]
figu.bar_plot_from_df(ax1, Change_AM_dt, **kwards_bar)
ax1.format(title='Partitioning (AM)', 
           ylim=(-85, 65), yticks=[-80, -45, -10, 25, 60], 
           ylabel='Changes (mm/AM)', )
ax1.legend(ncol=1, loc='lc', labelspacing=0.25)

ax2 = axes[1]
figu.bar_plot_from_df(ax2, Change_JJA_dt, **kwards_bar)
ax2.format(title='Partitioning (JJA)', 
           ylim=(-20, 130), yticks=[0, 30, 60, 90, 120], 
           ylabel='Changes (mm/JJA)', )
ax2.legend(ncol=1, loc='uc', labelspacing=0.25)

## ***********************
## * 相对贡献表格
## ***********************
aax1 = ax1.panel_axes('b', space='5em', width='10em', share=False)
aax1.format(xticks=[], yticks=[])
aax1.spines[:].set_visible(False)

cellText = np.concatenate([Contrib_AM.index.values.reshape(-1, 1), 
                           Contrib_AM.applymap(lambda x:f'{x:.1%}')], axis=1)

tax1 = aax1.table(cellText=cellText,
                  cellColours=[['blue3', 'blue3', 'blue3', 'blue3'], 
                               ['red3', 'red3', 'red3', 'red3'], 
                               ['grey3', 'grey3', 'grey3', 'grey3']], 
                  bbox=[0, 0, 1, 1], 
                #   rowLoc='center',
                  colLabels=['']+columns, 
                  colLoc='center', )

tax1.set(fontsize=14, animated=True, )

aax2 = ax2.panel_axes('b', space='5em', width='10em', share=False)
aax2.format(xticks=[], yticks=[])
aax2.spines[:].set_visible(False)

cellText = np.concatenate([Contrib_JJA.index.values.reshape(-1, 1), 
                           Contrib_JJA.applymap(lambda x:f'{x:.1%}')], axis=1)

tax2 = aax2.table(cellText=cellText,
                  cellColours=[['blue3', 'blue3', 'blue3', 'blue3'], 
                               ['red3', 'red3', 'red3', 'red3'], 
                               ['grey3', 'grey3', 'grey3', 'grey3']], 
                  bbox=[0, 0, 1, 1], 
                #   rowLoc='center',
                  colLabels=['']+columns, 
                  colLoc='center', )

tax2.set(fontsize=14, animated=True, )

fig.savefig('./pics/FIG9_定量贡献分解_PREC.png')
fig.show()

#%%