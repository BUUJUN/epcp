# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2022/11/18 11:26:31 
 
@author: BUUJUN WANG
"""
#%%
# 导入必要的库
import os
import torch
import torch.optim as optim
import optuna

import importlib
import buujun.functions as func
import buujun.models as models
importlib.reload(func)
importlib.reload(models)

# %%

study_name = 'CNN_R95d_SC_221118'

dir_cnn = '../data/cnn/'+study_name+'/'
if not os.path.exists(dir_cnn):
    os.mkdir(dir_cnn)

best_params = optuna.load_study(study_name=study_name, storage='sqlite:///'+study_name+'.db').best_params

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_loops", type=int, default=500, help="执行次数")
args = parser.parse_args()

# %%
def create_and_training(i):
    train_loader, test_loader, _, _ = func.data_loader(batch_size=best_params['batch_size'])
    net = models.ConvNN(channels=best_params['channels'], drop_out=best_params['drop_out']).to(device=models.device)
    optimizer = getattr(optim, best_params['optimizer'])(net.parameters(), lr=best_params['lr'])
    for epoch in range(models.args.n_epochs):
        func.train_loop(train_loader, net, optimizer)
        recall, precision = func.test_loop(test_loader, net)

        if recall >= 0.90 and precision >= 0.30:
            if epoch == models.args.n_epochs-1:
                path = dir_cnn+study_name+f'_{i:d}_{recall:.2f}_{precision:.2f}.pth'
            else:
                path = dir_cnn+study_name+f'_{recall:.2f}_{precision:.2f}_{epoch+1:d}.pth'
            torch.save(net, path)

    with open(dir_cnn+'trian.log.csv', 'a') as f:
        f.write(f'{i:d},{recall:.2f},{precision:.2f}\n')
        print(f'{i:d},{recall:.2f},{precision:.2f}')
    return None

if __name__ == "__main__":
    print(f'Training model: {study_name}')
    print(f'Best Parameter: {best_params}')
    print('index,recall,precision')
    if not os.path.exists(dir_cnn+'trian.log.csv'):
        with open(dir_cnn+'trian.log.csv', 'w') as f:
            f.write(f'Training model: {study_name} \n')
            f.write('index,recall,precision\n')
        start = 0
    else:
        with open(dir_cnn+'trian.log.csv', 'r') as f:
            lines = f.readlines()
        start = int(lines[-1].split(',')[0])+1

    [create_and_training(i) for i in range(start, start+args.n_loops)]