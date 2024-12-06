# -*- encoding: utf-8 R95d-*-
"""
Created on 2023/03/22 21:35:15 
 
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
cnn_res = pd.read_csv(para.model_result, index_col='time', 
                      parse_dates=['time'], date_parser=calc.date_parse)

ds = xr.open_dataset(para.prec_path).sel(time=slice('1961', '2018'))

# 筛选 4, 5, 6， 7， 8
cnn_res = cnn_res[np.isin(cnn_res.index.month, [4, 5, 6 ,7, 8])]
ds = ds.sel(time=ds.time[np.isin(ds.time.dt.month, [4, 5, 6 ,7, 8])])

#%%
days = ds.sel(time=ds.time.dt.year==2000).time.size
year = np.array(list(set(ds.time.dt.year.data)))
lon = ds.lon.data
lat = ds.lat.data

expr_3d = xr.where((ds.ep_day==1)|(np.isnan(ds.ep_day)), ds.prec, np.nan)

f_mean = np.empty((2, year.size))
f_dist = np.empty((2, year.size))
p_mean = np.empty((2, lat.size, lon.size, year.size))
p_dist = np.empty((2, lat.size, lon.size, year.size))

for i in [0, 1]:
    # P = p_0 * f_0 + p_1 * f_1
    pattern = cnn_res.index[cnn_res.predict_ep==i]

    p = para.fill_mask(expr_3d.loc[:, :, pattern].resample(time='Y').sum())\
        / xr.DataArray((cnn_res.predict_ep==i).resample('Y').sum()) # 对应该环流型对应的降水
    
    p_avg = p.mean(dim=['lon', 'lat'])
    
    f = (cnn_res.predict_ep==i).resample('Y').sum()  # 某环流型的个数

    f_mean[i] = f.mean()
    f_dist[i] = f - f_mean[i]
    p_mean[i] = xr.where(np.isnan(p), np.nan, p.mean(dim='time'))
    p_dist[i] = p - p_mean[i]

f_dist = np.where(np.isnan(f_dist), 0, f_dist)

p_mean_da = xr.DataArray(data=p_mean, dims=['pattern', 'lat', 'lon', 'time'], 
                         coords=dict(pattern=['NEPCP', 'EPCP'], lon=lon, lat=lat, time=year))
p_dist_da = p_mean_da.copy(data=p_dist)
f_mean_da = p_mean_da[:, 0, 0, :].copy(data=f_mean)
f_dist_da = p_mean_da[:, 0, 0, :].copy(data=f_dist)

therm = f_mean_da*p_dist_da
dynam = f_dist_da*p_mean_da
nolin = f_dist_da*p_dist_da # - (f_dist*p_dist).mean(axis=0)
total = therm + dynam + nolin

dt = pd.DataFrame(data={'total':  total.mean(dim=['lon', 'lat']).sum(dim='pattern'), 
                        'total_0':total[0].mean(dim=['lon', 'lat']), 
                        'total_1':total[1].mean(dim=['lon', 'lat']), 
                        'dynam':  dynam.mean(dim=['lon', 'lat']).sum(dim='pattern'), 
                        'dynam_0':dynam[0].mean(dim=['lon', 'lat']), 
                        'dynam_1':dynam[1].mean(dim=['lon', 'lat']), 
                        'therm':  therm.mean(dim=['lon', 'lat']).sum(dim='pattern'), 
                        'therm_0':therm[0].mean(dim=['lon', 'lat']), 
                        'therm_1':therm[1].mean(dim=['lon', 'lat']), 
                        'nolin':  nolin.mean(dim=['lon', 'lat']).sum(dim='pattern'), 
                        'nolin_0':nolin[0].mean(dim=['lon', 'lat']), 
                        'nolin_1':nolin[1].mean(dim=['lon', 'lat']), }, 
                  index=pd.date_range('1961-01-01', '2018-12-31', freq='y'))

def statics(series):
    slope=stats.linregress(x=year, y=series).slope
    pvalue=stats.linregress(x=year, y=series).pvalue
    late=series.loc[para.PL]
    early=series.loc[para.PE]
    delta=late.mean()-early.mean()
    ttest=stats.ttest_ind(a=late, b=early).pvalue
    return pd.Series(dict(slope=slope, pvalue=pvalue, delta=delta, ttest=ttest))
dt_statics = dt.apply(statics)

#%%
def pdf(data, bins):
    return pd.Series(stats.gaussian_kde(data)(bins))

# EPCP
xbin_epcp = 6
xticks_epcp = np.arange(0, 46, xbin_epcp)
xticks_epcp_pdf = np.linspace(0, 50, 100)
xticklabels_epcp = xticks_epcp[:-1].astype('str').tolist()
xbar_epcp = xticks_epcp[:-1] + xbin_epcp/2

prec_epcp = cnn_res.precipitation[cnn_res.predict_ep==1]

prec_epcp_hist_pe, _ = np.histogram(prec_epcp.loc[para.PE], bins=xticks_epcp)
prec_epcp_hist_pl, _ = np.histogram(prec_epcp.loc[para.PL], bins=xticks_epcp)
change_epcp = (prec_epcp_hist_pl - prec_epcp_hist_pe) / prec_epcp_hist_pe

prec_epcp_pdf_pe = pdf(prec_epcp.loc[para.PE], bins=xticks_epcp_pdf)
prec_epcp_pdf_pl = pdf(prec_epcp.loc[para.PL], bins=xticks_epcp_pdf)
change_epcp_pdf = (prec_epcp_pdf_pl - prec_epcp_pdf_pe) / prec_epcp_pdf_pe


# non-EPCP
xbin_nepcp = 3
xticks_nepcp = np.arange(0, 22, xbin_nepcp)
xticks_nepcp_pdf = np.linspace(0, 25, 100)
xticklabels_nepcp = xticks_nepcp[:-1].astype('str').tolist()
xbar_nepcp = xticks_nepcp[:-1] + xbin_nepcp/2

prec_nepcp = cnn_res.precipitation[cnn_res.predict_ep==0]

prec_nepcp_hist_pe, _ = np.histogram(prec_nepcp.loc[para.PE], bins=xticks_nepcp)
prec_nepcp_hist_pl, _ = np.histogram(prec_nepcp.loc[para.PL], bins=xticks_nepcp)
change_nepcp = (prec_nepcp_hist_pl - prec_nepcp_hist_pe) / prec_nepcp_hist_pe

prec_nepcp_pdf_pe = pdf(prec_nepcp.loc[para.PE], bins=xticks_nepcp_pdf)
prec_nepcp_pdf_pl = pdf(prec_nepcp.loc[para.PL], bins=xticks_nepcp_pdf)
change_nepcp_pdf = (prec_nepcp_pdf_pl - prec_nepcp_pdf_pe) / prec_nepcp_pdf_pe


# 画图
importlib.reload(figu)
fig, axes = pplt.subplots([[1, 3, 5, 5], 
                           [1, 3, 5, 5],  
                           [1, 3, 5, 5],  
                           [1, 3, 5, 5],  
                           [2, 4, 5, 5], 
                           [2, 4, 5, 5], 
                           [2, 4, 0, 0], 
                           [2, 4, 0, 0],], 
                          figsize=(12, 5), hspace=(0), wspace=(1, None, None), 
                          abcloc='ul')

axes.format(xminorticks=[], )
axes[:4].format(yminorticks=[])
axes[0:2].format(xlim=(0, xticks_epcp[-1]), )
axes[2:4].format(xlim=(0, xticks_nepcp[-1]), ytickloc='right', abcloc='ur')

axes[0].format(title='EPCPs', titleloc='l', xticks=[],
               ylabel='Counts', )
axes[2].format(title='NEPCPs', titleloc='r', xticks=[],)

axes[1].format(xticks=xticks_epcp[::1], xticklabels=xticklabels_epcp[::1], 
               xlabel='Precipitation (mm)', 
               ylim=(-0.4, 0.9), yticks=[0, 0.5, 1, 1.5], 
               yticklabels=['0%', '50%', '100%', '150%'], ylabel='Changes', )
axes[3].format(xticks=xticks_nepcp[::1], xticklabels=xticklabels_nepcp[::1], 
               xlabel='Precipitation (mm)', 
               ylim=(-0.4, 0.9), yticks=[0, 0.5, 1, 1.5], 
               yticklabels=['0%', '50%', '100%', '150%'])

axes[0].spines.bottom.set_linewidth(1)
axes[2].spines.bottom.set_linewidth(1)
axes[1].spines.top.set_linewidth(1)
axes[3].spines.top.set_linewidth(1)

## ***********************
## * 变化
## ***********************
color_pe = 'indigo5'
color_pl = 'orange5'
label_pe = 'PE'
label_pl = 'PL'
width = 0.8

axes[0].bar(xbar_epcp-xbin_epcp*(width/2)*0.5, prec_epcp_hist_pe, width=width/2, colors=color_pe)
axes[0].bar(xbar_epcp+xbin_epcp*(width/2)*0.5, prec_epcp_hist_pl, width=width/2, colors=color_pl)
axes[2].bar(xbar_nepcp-xbin_nepcp*(width/2)*0.5, prec_nepcp_hist_pe, width=width/2, colors=color_pe, label=label_pe)
axes[2].bar(xbar_nepcp+xbin_nepcp*(width/2)*0.5, prec_nepcp_hist_pl, width=width/2, colors=color_pl, label=label_pl)
axes[2].legend(ncol=1, loc='cr')

# axes[1].bar(xbar_epcp, change_epcp, width=width, negpos=True)
# axes[1].axhline(0, color='k', linewidth=1)
# axes[3].bar(xbar_nepcp, change_nepcp, width=width, negpos=True)
# axes[3].axhline(0, color='k', linewidth=1)
alpha = 1
axes[1].plot(xticks_epcp_pdf, change_epcp_pdf, color='k')
axes[1].axhline(0, color='k', linewidth=1)
axes[1].fill_between(xticks_epcp_pdf, 0, change_epcp_pdf, where=change_epcp_pdf>0, 
                     facecolor='red5', alpha=alpha)
axes[1].fill_between(xticks_epcp_pdf, change_epcp_pdf, 0, where=change_epcp_pdf<0, 
                     facecolor='blue5', alpha=alpha)

axes[3].plot(xticks_nepcp_pdf, change_nepcp_pdf, color='k')
axes[3].axhline(0, color='k', linewidth=1)
axes[3].fill_between(xticks_nepcp_pdf, 0, change_nepcp_pdf, where=change_nepcp_pdf>0, 
                     facecolor='red5', alpha=alpha)
axes[3].fill_between(xticks_nepcp_pdf, change_nepcp_pdf, 0, where=change_nepcp_pdf<0, 
                     facecolor='blue5', alpha=alpha)

## ***********************
## * 分解
## ***********************
delta = dt_statics.loc['delta']

labels = ['Total', 'Dyn', 'Thermo', 'Inter']
labels_long = ['Total', 'Dynamic', 'Thermodynamic', 'Interaction']
colors = ['orange', 'blue7', 'red7', 'grey']

delta_dt = pd.DataFrame(dict(All=pd.Series(delta[0::3].values, index=labels), 
                             NEPCPs=pd.Series(delta[1::3].values, index=labels),
                             EPCPs=pd.Series(delta[2::3].values, index=labels),))

contribute_dt = delta_dt.iloc[1:] / delta_dt.iloc[0]

figu.bar_plot_from_df(axes[4], delta_dt, colors=colors, width=1, legend=False)
axes[4].format(title='Quantitative partitioning', titleloc='left', abcloc='l', 
               ylim=(-30, 100), yticks=[-20, 0, 20, 40, 60, 80], 
               yticklabels=['-20', '0', '20', '40', '60', '80'], 
               ylabel='Changes (mm/AMJJA)', )
axes[4].legend(ncol=1, loc='uc')

## ***********************
## * 相对贡献
## ***********************
aax = fig.add_axes(rect=[axes[4].get_position().x0, 0.03, 
                         axes[4].get_position().width, axes[4].get_position().height/2.5], )
aax.format(xticks=[], yticks=[])
aax.spines[:].set_visible(False)

cellText = np.concatenate([contribute_dt.index.values.reshape(-1, 1), 
                           contribute_dt.applymap(lambda x:format(x, '.2%'))], axis=1)

tax = aax.table(cellText=cellText,
                cellColours=[['blue3', 'blue3', 'blue3', 'blue3'], 
                             ['red3', 'red3', 'red3', 'red3'], 
                             ['grey3', 'grey3', 'grey3', 'grey3']], 
                bbox=[0, 0, 1, 1], 
                colLabels=['']+contribute_dt.columns.to_list(), 
                colLoc='center')

tax.set(fontsize=14)

print(f'''
      极端降水变化: {delta_dt.values[0, 0]:.2f} mm/AMJJA
      极端降水变化中 EPCPs 的占比: {delta_dt.values[0, 2]/delta_dt.values[0, 0]*100:.2f} %
      极端降水变化的热力贡献: {delta_dt.values[2, 0]/delta_dt.values[0, 0]*100:.2f} %
''')

fig.savefig('./pics/FIG9_量化贡献_全年.png')
pplt.show()

#%%
# 作图
kwards_geo = dict(lonlat=[108, 120, 21, 26.5], 
                  lonticks=np.array([110, 115, 120]), 
                  latticks=np.array([21, 23, 25]),)

def format_plot(ax, title='', unit=''):
    figu2.geo_format(ax, **kwards_geo)
    ax.set_title(title, loc='left')
    ax.set_title(unit, loc='right')
    # ax.contour(topo.lon, topo.lat, topo_mask, levels=[1], colors='g')

units_abs = ['mm/AMJJA', 'mm/AMJJA']
units_rla = ['%', '%']
cmaps = ['anomaly', 'anomaly']

titles = ['Therm EPCP', 'Dynam EPCP']
levels_rla = [np.linspace(-250, 250, 11),
              np.linspace(-50, 50, 11)]

levels_abs = [np.linspace(-100, 100, 11),
              np.linspace(-20, 20, 11)]

diff_cal = lambda dataarray: \
    dataarray.loc[para.PL].mean(dim='time') - dataarray.loc[para.PE].mean(dim='time')

diff_t = diff_cal(total[1]).mean(dim=['lon', 'lat'])
diff_p = [diff_cal(therm[1]), diff_cal(dynam[1])]
ctri_p = [i/diff_t*100 for i in diff_p]

# absolute
importlib.reload(figu2)
fig, axes = pplt.subplots([[1], [2]], figsize=(6.7, 6), proj='cyl')

for i, ax in enumerate(axes):
    format_plot(ax, title=titles[i], unit=units_abs[i])
    cf = figu2.contourf_plot(ax, lon.data, lat.data, 
                             diff_p[i], 
                             cmap=cmaps[i], 
                             levels=levels_abs[i],
                             extend='both', )
    ax.colorbar(cf, loc='right', space=-1)

# # relative
# importlib.reload(figu2)
# fig, axes = pplt.subplots([[1], [2]], figsize=(6.7, 6), proj='cyl')

# for i, ax in enumerate(axes):
#     format_plot(ax, title=titles[i], unit=units_rla[i])
#     cf = figu2.contourf_plot(ax, lon.data, lat.data, 
#                              ctri_p[i], 
#                              cmap=cmaps[i], 
#                              levels=levels_rla[i],
#                              extend='both', )
#     ax.colorbar(cf, loc='right')

# %%
