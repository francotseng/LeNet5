#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2020, Tseng. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.

import torch.nn as nn

from abc import ABC


class LeNet5(nn.Module, ABC):
    """
    略微修改过的 LeNet5 模型

    Attributes:
        need_dropout (bool): 是否需要增加随机失活层
        conv1 (nn.Conv2d): 卷积核1，默认维度 (6, 5, 5)
        pool1 (nn.MaxPool2d): 下采样函数1，维度 (2, 2)
        conv2 (nn.Conv2d): 卷积核2，默认维度 (16, 5, 5)
        pool2 (nn.MaxPool2d): 下采样函数2，维度 (2, 2)
        conv3 (nn.Conv2d): 卷积核3，默认维度 (120, 5, 5)
        fc1 (nn.Linear): 全连接函数1，维度 (120, 84)
        fc2 (nn.Linear): 全连接函数2，维度 (84, 10)
        dropout (nn.Dropout): 随机失活函数
    """
    def __init__(self, dropout_prob=0., halve_conv_kernels=False):
        """
        初始化模型各层函数
        :param dropout_prob: 随机失活参数
        :param halve_conv_kernels: 是否将卷积核数量减半
        """
        super(LeNet5, self).__init__()
        kernel_nums = [6, 16]
        if halve_conv_kernels:
            kernel_nums = [num // 2 for num in kernel_nums]
        self.need_dropout = dropout_prob > 0

        # 卷积层 1，6个 5*5 的卷积核
        # 由于输入图像是 28*28，所以增加 padding=2，扩充到 32*32
        self.conv1 = nn.Conv2d(1, kernel_nums[0], (5, 5), padding=2)
        # 下采样层 1，采样区为 2*2
        self.pool1 = nn.MaxPool2d((2, 2))
        # 卷积层 2，16个 5*5 的卷积核
        self.conv2 = nn.Conv2d(kernel_nums[0], kernel_nums[1], (5, 5))
        # 下采样层 2，采样区为 2*2
        self.pool2 = nn.MaxPool2d((2, 2))
        # 卷积层 3，120个 5*5 的卷积核
        self.conv3 = nn.Conv2d(kernel_nums[1], 120, (5, 5))
        # 全连接层 1，120*84 的全连接矩阵
        self.fc1 = nn.Linear(120, 84)
        # 全连接层 2，84*10 的全连接矩阵
        self.fc2 = nn.Linear(84, 10)
        # 随机失活层，失活率为 dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        前向传播函数，返回给定输入数据的预测标签数组
        :param x: 维度为 (batch_size, 28, 28) 的图像数据
        :return: 维度为 (batch_size, 10) 的预测标签
        """
        x = x.unsqueeze(1)                      # (batch_size, 1, 28, 28)
        feature_map = self.conv1(x)             # (batch_size, 6, 28, 28)
        feature_map = self.pool1(feature_map)   # (batch_size, 6, 14, 14)
        feature_map = self.conv2(feature_map)   # (batch_size, 16, 10, 10)
        feature_map = self.pool2(feature_map)   # (batch_size, 16, 5, 5)
        feature_map = self.conv3(feature_map).squeeze()     # (batch_size, 120)
        out = self.fc1(feature_map)             # (batch_size, 84)
        if self.need_dropout:
            out = self.dropout(out)             # (batch_size, 10)
        out = self.fc2(out)                     # (batch_size, 10)
        return out
