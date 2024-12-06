# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/05/15 22:15:48 
 
@author: BUUJUN WANG
"""
#%%
import xarray as xr
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn import metrics

import importlib
import buujun.parameters as para
import buujun.models as models
importlib.reload(para)
importlib.reload(models)


def data_loader(batch_size=models.args.batch_size):
    # 数据读取：ep/slp/z
    slp = xr.open_dataset(para.var_path['msl']).msl.sel(longitude=para.lon_cnn, latitude=para.lat_cnn)
    zg = xr.open_dataset(para.var_path['z']).z.sel(longitude=para.lon_cnn, latitude=para.lat_cnn, level=500)
    ds_prec = xr.open_dataset(para.prec_path)
    # 标准化
    slp_normal = (slp - slp.mean(dim=['time'])) / slp.std(dim=['time'])
    zg_normal = (zg - zg.mean(dim=['time'])) / zg.std(dim=['time'])
    del slp, zg
    # dataarray -> tensor
    features = torch.tensor(np.stack([slp_normal.data, zg_normal.data], axis=1), 
                            dtype=torch.float, requires_grad=False)
    labels = torch.tensor(ds_prec.ep_day_sc.data, 
                          dtype=torch.long, requires_grad=False)
    # 划分训练集和验证集
    train_idx, test_idx = torch.utils.data.random_split(
        np.arange(features.shape[0]), 
        np.round(np.array([0.7, 0.3])*features.shape[0]).astype('int'), 
        generator=torch.Generator().manual_seed(0)
    )
    trainset = torch.utils.data.TensorDataset(features[list(train_idx)], labels[list(train_idx)])
    testset = torch.utils.data.TensorDataset(features[list(test_idx)], labels[list(test_idx)])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)
    return train_loader, test_loader, train_idx, test_idx

def train_loop(train_loader, net, optimizer):     # 训练 loop
    net.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([labels.detach().sum(), (labels.detach()==0).sum()], dtype=torch.float).to(device=models.device))
        inputs = inputs.to(device=models.device)
        labels = labels.to(device=models.device)
        optimizer.zero_grad()
        outputs = net(inputs) # 训练
        loss = criterion(outputs, labels) # 损失函数
        loss.backward() # 后向传播
        optimizer.step() # 优化
    return None

def test_loop(test_loader, net):     # 测试 loop
    net.eval()
    true_list = list(); pred_list = list()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device=models.device)
            outputs = net(inputs)
            _, predicts = torch.max(outputs.data, dim=1)  # max_value, max_index = torch.max(outputs.data, dim=1)
            pred_list.append(predicts.cpu())
            true_list.append(labels)
    recall = metrics.recall_score(torch.hstack(true_list), torch.hstack(pred_list), zero_division=0)
    precision = metrics.precision_score(torch.hstack(true_list), torch.hstack(pred_list), zero_division=0)
    return recall, precision