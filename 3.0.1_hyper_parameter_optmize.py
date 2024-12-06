# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2022/10/30 14:26:43 
 
@author: BUUJUN WANG
"""
# %%
import xarray as xr
import torch
import optuna
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn import metrics

import importlib
import buujun.models as models
import buujun.parameters as para
importlib.reload(models)
importlib.reload(para)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="训练次数")
parser.add_argument("--n_trials", type=int, default=100, help="超参优化次数")
args = parser.parse_args()

torch.set_num_threads(48)

# 数据读取：ep/slp/z
slp = xr.open_dataset(para.var_path['msl']).msl.sel(longitude=para.lon_cnn, latitude=para.lat_cnn)
zg = xr.open_dataset(para.var_path['z']).z.sel(longitude=para.lon_cnn, latitude=para.lat_cnn, level=500)
ds_prec = xr.open_dataset(para.prec_path)

# 标准化
slp_normal = (slp - slp.mean(dim=['time'])) / slp.std(dim=['time'])
zg_normal = (zg - zg.mean(dim=['time'])) / zg.std(dim=['time'])
del slp, zg

#%%
# dataarray -> tensor
features = torch.tensor(np.stack([slp_normal.data, zg_normal.data], axis=1), 
                        dtype=torch.float, requires_grad=False)
del slp_normal, zg_normal
labels = torch.tensor(ds_prec.ep_day_sc.data, dtype=torch.long, requires_grad=False)

# 划分训练集和验证集
train_idx, test_idx = torch.utils.data.random_split(
    np.arange(features.shape[0]), 
    np.round(np.array([0.7, 0.3])*features.shape[0]).astype('int'), 
    generator=torch.Generator().manual_seed(0)
)
trainset = torch.utils.data.TensorDataset(features[list(train_idx)], labels[list(train_idx)])
testset = torch.utils.data.TensorDataset(features[list(test_idx)], labels[list(test_idx)])

# %%
def data_loader(trainset, testset, batch_size=models.args.batch_size):
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)
    return train_loader, test_loader

# %%
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

def optuna_objective(trial):   # 定义进行一次 trail 的过程
    batch_size = trial.suggest_int("batch_size", 60, 300, step=60)
    channels = trial.suggest_int("channels", 4, 20, step=4)
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    drop_out = trial.suggest_float("drop_out", 0, 0.5, step=0.1)
    # kernel_size = trial.suggest_int("kernel_size", 3, 5, step=1)
    kernel_size = 5

    train_loader, test_loader = data_loader(trainset, testset, batch_size=batch_size)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    
    fn = []
    for l in range(3):
        net = models.ConvNN(channels=channels, kernel_size=kernel_size, drop_out=drop_out).to(device=models.device)
        optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
        for epoch in range(args.n_epochs):
            train_loop(train_loader, net, optimizer)
            recall, precision = test_loop(test_loader, net)
            if recall==0.0 or precision==0.0:
                f1 = 0
            else:
                f1 = 2*recall*precision/(0.3*recall+1.7*precision)
            if epoch % (args.n_epochs//5) == 0:
                print(f'[{epoch}/{models.args.n_epochs}]: recall={recall:.3f}, precision={precision:.3f}, f1={f1:.3f}')
        fn.append(f1)
        trial.report(f1, l)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return np.mean(np.array(fn))

#%%
def f1_distribution():
    import matplotlib.pyplot as plt
    import sys

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    recall = np.linspace(.1,1,10); precision = np.linspace(.1,1,10)
    m_recall, m_precision = np.meshgrid(recall, precision)
    f = 2*m_recall*m_precision/(0.3*m_recall+1.7*m_precision)
    ax.pcolormesh(m_recall, m_precision, f, edgecolor='k', cmap='tab20b')
    ax.set(
        title = 'f1', 
        xlabel='recall', 
        xticks=recall, xticklabels=np.around(recall, 1),
        ylabel='precision', 
        yticks=precision, yticklabels=np.around(precision, 1)
    )
    for i, r in enumerate(recall):
        for j, p in enumerate(precision):
            text = ax.text(r, p, f'{f[j, i]:.3f}',ha="center", va="center", color="k")
    plt.show()
    return None 

# %%
if __name__ == "__main__":

    study_name = para.study_name

    print(f'Optimize model: '+study_name)

    study = optuna.create_study(study_name=study_name, storage='sqlite:///'+study_name+'.db', direction='maximize', load_if_exists=True)

    # study.optimize(func=optuna_objective, n_trials=10)
    study.optimize(func=optuna_objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
