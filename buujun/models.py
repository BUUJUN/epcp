# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2022/11/16 14:48:38 
 
@author: BUUJUN WANG
"""
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np

#%%
## 超参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="训练次数")
parser.add_argument("--batch_size", type=int, default=40, help="分批大小")
parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
parser.add_argument("--optimizer", type=str, default='SGD', help="优化器")
parser.add_argument("--ngpu", type=int, default=0, help="使用的 GPU 的个数")
## 卷积层的参数
parser.add_argument("--kernel_size", type=int, default=4, help="卷积核大小")
parser.add_argument("--stride", type=int, default=1, help="卷积核移动的步幅")
parser.add_argument("--padding", type=int, default=0, help="卷积核周围空白填充量")
## 全连接层的参数
parser.add_argument("--drop_out", type=float, default=0.5, help="剪枝的概率")
## 网络层数的参数
parser.add_argument("--channels", type=int, default=16, help="剪枝的概率")
# args = parser.parse_args()
args = parser.parse_args(args=[])
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
optimizer = getattr(optim, args.optimizer)

#%%
class ConvNN(nn.Module):
    def __init__(
        self, 
        channels=args.channels, 
        kernel_size=args.kernel_size, 
        stride=args.stride, 
        padding=args.padding, 
        drop_out=args.drop_out
    ):
        super(ConvNN, self).__init__()
        self.ngpu = args.ngpu
        
        def conv_layer(in_channels, out_channels, maxpool=True, groups=1):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
                nn.ReLU(inplace=True),
            ]
            if maxpool: layers.append(nn.MaxPool2d(2, 2))
            return layers

        def linear_layer(in_channels, out_channels):
            layers = [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_out),
            ]
            return layers
        
        def ctol(img_size=31, maxpool=True):
            if maxpool:
                return np.floor((img_size+1-kernel_size)/2).astype('int')
            return img_size+1-kernel_size

        self.main = nn.Sequential(
            *conv_layer(2, channels*2, groups=2),
            *conv_layer(channels*2, channels*4),
            nn.Flatten(),
            *linear_layer(channels*4*(ctol(ctol(31))**2), channels*8),
            *linear_layer(channels*8, channels*4),
            nn.Linear(channels*4, 2)
        )
    
    def forward(self, input):
        return self.main(input)
    
class ConvNN_4var(nn.Module):
    def __init__(
        self, 
        channels=args.channels, 
        kernel_size=args.kernel_size, 
        stride=args.stride, 
        padding=args.padding, 
        drop_out=args.drop_out
    ):
        super(ConvNN_4var, self).__init__()
        self.ngpu = args.ngpu
        
        def conv_layer(in_channels, out_channels, maxpool=True, groups=1):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
                nn.ReLU(inplace=True),
            ]
            if maxpool: layers.append(nn.MaxPool2d(2, 2))
            return layers

        def linear_layer(in_channels, out_channels):
            layers = [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_out),
            ]
            return layers
        
        def ctol(img_size=31, maxpool=True):
            if maxpool:
                return np.floor((img_size+1-kernel_size)/2).astype('int')
            return img_size+1-kernel_size

        self.main = nn.Sequential(
            *conv_layer(4, channels*4, groups=4),
            *conv_layer(channels*4, channels*4),
            nn.Flatten(),
            *linear_layer(channels*4*(ctol(ctol(31))**2), channels*8),
            *linear_layer(channels*8, channels*4),
            nn.Linear(channels*4, 2)
        )
    
    def forward(self, input):
        return self.main(input)
    

class ConvNN_3var(nn.Module):
    def __init__(
        self, 
        channels=args.channels, 
        kernel_size=args.kernel_size, 
        stride=args.stride, 
        padding=args.padding, 
        drop_out=args.drop_out
    ):
        super(ConvNN_3var, self).__init__()
        self.ngpu = args.ngpu
        
        def conv_layer(in_channels, out_channels, maxpool=True, groups=1):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
                nn.ReLU(inplace=True),
            ]
            if maxpool: layers.append(nn.MaxPool2d(2, 2))
            return layers

        def linear_layer(in_channels, out_channels):
            layers = [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_out),
            ]
            return layers
        
        def ctol(img_size=31, maxpool=True):
            if maxpool:
                return np.floor((img_size+1-kernel_size)/2).astype('int')
            return img_size+1-kernel_size

        self.main = nn.Sequential(
            *conv_layer(3, channels*3, groups=3),
            *conv_layer(channels*3, channels*4),
            nn.Flatten(),
            *linear_layer(channels*4*(ctol(ctol(31))**2), channels*8),
            *linear_layer(channels*8, channels*4),
            nn.Linear(channels*4, 2)
        )
    
    def forward(self, input):
        return self.main(input)
    

class ConvNN_1var(nn.Module):
    def __init__(
        self, 
        channels=args.channels, 
        kernel_size=args.kernel_size, 
        stride=args.stride, 
        padding=args.padding, 
        drop_out=args.drop_out
    ):
        super(ConvNN_1var, self).__init__()
        self.ngpu = args.ngpu
        
        def conv_layer(in_channels, out_channels, maxpool=True, groups=1):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
                nn.ReLU(inplace=True),
            ]
            if maxpool: layers.append(nn.MaxPool2d(2, 2))
            return layers

        def linear_layer(in_channels, out_channels):
            layers = [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_out),
            ]
            return layers
        
        def ctol(img_size=31, maxpool=True):
            if maxpool:
                return np.floor((img_size+1-kernel_size)/2).astype('int')
            return img_size+1-kernel_size

        self.main = nn.Sequential(
            *conv_layer(1, channels*2),
            *conv_layer(channels*2, channels*4),
            nn.Flatten(),
            *linear_layer(channels*4*(ctol(ctol(31))**2), channels*8),
            *linear_layer(channels*8, channels*4),
            nn.Linear(channels*4, 2)
        )
    
    def forward(self, input):
        return self.main(input)
    

def LRP(features, net):
    from zennit.composites import layer_map_base, LayerMapComposite, NameMapComposite, EpsilonGammaBox, EpsilonPlusFlat, SpecialFirstLayerMapComposite
    from zennit.rules import Epsilon, AlphaBeta, ZPlus, Norm, Pass, Flat
    from zennit.types import Convolution, Linear, Activation, AvgPool, BatchNorm
    from zennit.canonizers import SequentialMergeBatchNorm
    import torch
    layer_map = layer_map_base()+[
        # (Activation, Pass()),  # ignore activations
        # (AvgPool, Norm()),  # normalize relevance for any AvgPool
        # (Convolution, AlphaBeta(alpha=1, beta=0)),  # any convolutional layer
        (Convolution, ZPlus(zero_params='bias')),  # any convolutional layer
        (torch.nn.Linear, AlphaBeta(alpha=1, beta=0)),  # this is the dense Linear, not any
        # (torch.nn.Linear, Epsilon()),  # this is the dense Linear, not any
    ]
    canonizer = SequentialMergeBatchNorm()
    composite = LayerMapComposite(layer_map=layer_map, canonizers=[canonizer])
    # composite = SpecialFirstLayerMapComposite(layer_map=layer_map, first_map=[(Convolution, Flat()),],)
    # composite = EpsilonGammaBox(
    #     low=-3., high=3., 
    #     layer_map=[(BatchNorm, Pass()),(Linear, ZPlus()),],
    #     # first_map=[(Convolution, Flat()),],
    # )
    features.requires_grad = True
    net.eval()

    with composite.context(net) as modified_net:
        output = modified_net(features)
        relevance,  = torch.autograd.grad(
            output, features, grad_outputs=torch.ones_like(output)
        )

    return relevance

# %%
