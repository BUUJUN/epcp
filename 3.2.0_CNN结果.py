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
var1 = xr.open_dataset(para.var_path['u']).u.sel(longitude=para.lon_cnn, latitude=para.lat_cnn, level=850)
var2 = xr.open_dataset(para.var_path['v']).v.sel(longitude=para.lon_cnn, latitude=para.lat_cnn, level=850)
var3 = xr.open_dataset(para.var_path['z']).z.sel(longitude=para.lon_cnn, latitude=para.lat_cnn, level=500)
# 标准化
var1_normal = (var1 - var1.mean(dim=['time'])) / var1.std(dim=['time'])
var2_normal = (var2 - var2.mean(dim=['time'])) / var2.std(dim=['time'])
var3_normal = (var3 - var3.mean(dim=['time'])) / var3.std(dim=['time'])
del var1, var2, var3

#%%
# dataarray -> tensor
features = torch.tensor(np.stack([var1_normal.data, var2_normal.data, var3_normal.data], axis=1), 
                        dtype=torch.float, requires_grad=False)
del var1_normal, var2_normal, var3_normal

ds_prec = xr.open_dataset(para.prec_path)
labels = torch.tensor(ds_prec.ep_day_sc.data, dtype=torch.long, requires_grad=False)

## 卷积网络
net_path = para.model_path
net = torch.load(net_path).cpu()

## CNN 输入输出
with torch.no_grad():
    net.eval()
    outputs = net(features)
    probability = nn.functional.softmax(outputs, dim=1)
    _, predicts = torch.max(outputs.data, dim=1)  # 预测 0 还是 1

df = pd.DataFrame(dict(
    i=np.arange(ds_prec.time.size), 
    time=ds_prec.time, 
    precipitation=ds_prec.prec.mean(dim=['lon', 'lat']), 
    true_ep=ds_prec.ep_day_sc, 
    predict_ep=predicts.detach().numpy(), 
    probability=probability.detach().numpy()[:, 1]), index=ds_prec.time)

df.head()

#%%
df.to_csv(para.model_result, index=False)
os.system(f'chmod 400 {para.model_result}')
# %%
