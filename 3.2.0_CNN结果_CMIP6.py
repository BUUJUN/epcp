# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2022/05/05 11:55:16 
 
@author: 王伯俊
"""
#%%
import importlib
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mlp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy import stats
from sklearn import metrics
import torch
import torchvision
import torch.nn as nn
import optuna
import buujun.parameters as para
import buujun.models as models
importlib.reload(para)
importlib.reload(models)
import warnings
warnings.filterwarnings('ignore')

#%%
## 数据准备
path = '/home/yangsong3/wangbj/epcp_cmip6/data/cmip6/'
fn_var = 'vars_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_19600101-20141231_AMJJA_80-140_40-0.nc'
var_ds = xr.open_dataset(path+fn_var).sel(lon=para.lon_cnn, lat=para.lat_cnn_r)
var1 = var_ds.ua.sel(plev=850)
var2 = var_ds.va.sel(plev=850)
var3 = var_ds.zg.sel(plev=500)

# 标准化
var1_normal = (var1 - var1.mean(dim=['time'])) / var1.std(dim=['time'])
var2_normal = (var2 - var2.mean(dim=['time'])) / var2.std(dim=['time'])
var3_normal = (var3 - var3.mean(dim=['time'])) / var3.std(dim=['time'])
# del var1, var2, var3
# dataarray -> tensor
features = torch.tensor(np.stack([var1_normal.data, var2_normal.data, var3_normal.data], axis=1), 
                        dtype=torch.float, requires_grad=False)
# del var1_normal, var2_normal, var3_normal

# ds_prec = xr.open_dataset(para.prec_path)
fn_repe = 'repe_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_19600101-20141231_AMJJA_80-140_40-0.nc'
repe = xr.open_dataset(path+fn_repe).repe
labels = torch.tensor(repe.data, dtype=torch.long, requires_grad=False)

## 卷积网络
net_path = para.model_path
net = torch.load(net_path).cpu()

## CNN 输入输出
with torch.no_grad():
    net.eval()
    outputs = net(features)
    probability = nn.functional.softmax(outputs, dim=1)
    _, predicts = torch.max(outputs.data, dim=1)  # 预测 0 还是 1

cn05_mask = xr.open_dataset('../data/CN05.1/CN05.1_Mask_1961_2018_daily_01x01.nc').mask
mask_data = lambda array: array.where(cn05_mask.sel(lat=array.lat, lon=array.lon))

df = pd.DataFrame(dict(
    i=np.arange(repe.time.size), 
    time=repe.time, 
    precipitation=mask_data(var_ds.pr.sel(lon=para.lon_prec, lat=para.lat_prec)).mean(dim=['lon', 'lat']), 
    true_ep=repe, 
    predict_ep=predicts.detach().numpy(), 
    probability=probability.detach().numpy()[:, 1]), index=repe.time)

df.head()

#%%
df.to_csv('./result_hpo_uv850_z_0019_0429.pth_BCC-CSM2-MR.csv', index=False)
os.system(f'chmod 400 ./result_hpo_uv850_z_0019_0429.pth_BCC-CSM2-MR.csv')
# %%
