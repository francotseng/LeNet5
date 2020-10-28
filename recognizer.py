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

import os
import torch
import torch.nn as nn
import numpy as np

from sklearn import metrics

from model import LeNet5
from dataloader import load_data, show_image


class Recognizer(object):
    """
    手写数字识别器，内部使用 LeNet5 模型

    Attributes:
        data_path (dict)：数据集路径，包括数据目录、训练集和测试集文件名
        batch_size (int)：一个训练批次的数据量
        epochs (int)：训练的最大迭代次数，完整使用完一次训练数据为一个迭代
        dropout_prob (float)：随机失活参数
        halve_conv_kernels (bool)：是否将卷积核减半
        has_cuda：是否能够使用 cuda
        train_loader：训练集的数据加载器
        test_loader：测试集的数据加载器
        model：LeNet5 模型
        output_dir：输出模型文件的保存目录
        mode_path：输出模型文件的完整文件名
    """
    def __init__(self, data_path, batch_size=128, epochs=100,
                 dropout_prob=0, halve_conv_kernels=False):
        """
        初始化属性
        :param data_path: 数据集目录
        :param batch_size: 每个批次的数据数量
        :param epochs: 训练的最大迭代次数
        :param dropout_prob: 随机失活参数
        :param halve_conv_kernels: 是否将卷积核减半
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_prob = dropout_prob
        self.halve_conv_kernels = halve_conv_kernels

        self.has_cuda = torch.cuda.is_available()
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.output_dir = 'output/'

        # 生成模型文件名
        name = 'lenet5'
        if halve_conv_kernels:
            name += '_halve'
        if dropout_prob > 0:
            name += '_' + str(dropout_prob)
        self.model_path = os.path.join(
            self.output_dir, name+'.mdl')

        print('<Recognizer>: [batch_size] %d, [dropout_prob] %.1f, '
              '[halve_conv_kernel] %s'
              % (batch_size, dropout_prob, halve_conv_kernels))

    def prepare(self):
        """
        加载训练集和测试集数据
        :return:
        """
        self.train_loader = load_data(
            self.data_path['data_dir'], self.data_path['train_image'],
            self.data_path['train_label'], self.batch_size)
        self.test_loader = load_data(
            self.data_path['data_dir'], self.data_path['test_image'],
            self.data_path['test_label'], self.batch_size)

    def train(self):
        """
        模型训练，如果模型文件已存在则跳过训练，直接加载模型文件
        :return:
        """
        self.model = LeNet5(dropout_prob=self.dropout_prob,
                            halve_conv_kernels=self.halve_conv_kernels)
        if self.has_cuda:
            self.model.cuda()

        # 模型文件已经存在，直接加载模型后返回
        if os.path.exists(self.model_path):
            print('Train: model file exists, skip training.')
            print('Train: loading model state from file [%s] ...'
                  % self.model_path)
            self.model.load_state_dict(torch.load(self.model_path))
            return

        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=1e-4, betas=(0.9, 0.99))
        idx = 0                     # batch 计数
        is_stop = False             # 是否停止训练
        best_loss = float('inf')    # 最优评估 loss 值
        best_acc = 0.               # 最优评估准确率
        best_batch_idx = 0          # 最优批次
        best_model_state = None     # 模型最优状态

        print('Train: start training model ...')
        self.model.train()
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(self.train_loader):
                idx += 1
                if self.has_cuda:
                    x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()           # 梯度清零
                output = self.model(x)          # 前向传播计算预测值
                loss = criterion(output, y)     # 计算 loss
                loss.backward()                 # 反向传播计算梯度
                optimizer.step()                # 参数调整

                # 每 100 批次输出一次效果
                if idx % 100 == 0:
                    y = y.cpu()
                    y_pred = output.argmax(dim=1).cpu()
                    acc = metrics.accuracy_score(y, y_pred)  # 本轮训练准确率
                    # 使用测试集评估模型的准确率和 loss 值
                    eval_acc, eval_loss = self.eval(self.test_loader)
                    # 设置为 train 模式，因为在 eval() 中置为了 eval 模式
                    self.model.train()
                    suffix = ''
                    # 更新最优状态
                    if eval_loss < best_loss or eval_acc > best_acc:
                        suffix = ' *'
                        best_batch_idx = idx
                        best_loss = min(best_loss, eval_loss)
                        best_acc = max(best_acc, eval_acc)
                        best_model_state = self.model.state_dict()
                    msg = 'Train [Epoch {:>3}]: \tTrain Loss: {:7.3f}\t' + \
                          'Train Acc: {:>5.2%}\t' + \
                          'Eval Loss: {:7.3f}\tEval Acc: {:>5.2%}{}'
                    print(msg.format(epoch+1, loss.item(),
                                     acc, eval_loss, eval_acc, suffix))
                    # 如果超过连续 1000 个批次没有优化，则结束训练
                    if idx - best_batch_idx > 1000:
                        print('no optimization for more than 1000 batches, '
                              'auto stop training.')
                        is_stop = True
                        break
                if is_stop:
                    break
        print('Train: end training model, best loss {:.3f}, best acc {:.2%}'.
              format(best_loss, best_acc))
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        print('Train: saving model [%s] ...' % self.model_path)
        # 保存模型
        torch.save(best_model_state, self.model_path)
        # 加载模型最优状态
        self.model.load_state_dict(best_model_state)

    def eval(self, data_loader, is_test=False):
        """
        模型评估
        :param data_loader: 数据加载器
        :param is_test: 是否为测试模式
        :return: 在输入数据上的预测准确率和平均 loss 值
        """
        self.model.eval()
        wrong_cnt = 0                           # 统计预测错误数量
        loss_total = 0                          # 所有 loss 之和
        labels_all = np.array([], dtype=int)    # 真实标签列表
        predict_all = np.array([], dtype=int)   # 预测标签列表
        criterion = nn.CrossEntropyLoss(reduction='sum')

        for x, y in data_loader:
            if self.has_cuda:
                x, y = x.cuda(), y.cuda()
            output = self.model(x)              # 前向传播计算预测值
            loss = criterion(output, y)         # 计算 loss
            loss_total += loss
            y_pred = output.argmax(dim=1)       # 取出预测最大下标
            labels_all = np.append(labels_all, y.cpu())
            predict_all = np.append(predict_all, y_pred.cpu())

            if is_test:
                for i in range(len(y)):
                    if y[i] != y_pred[i]:
                        wrong_cnt += 1          # 统计预测错误数量
                        show_image(x[i].cpu(), 'label:%d, pred:%d'
                                   % (y[i], y_pred[i]))
                        print('wrong predict: label %d, predict %d'
                              % (y[i], y_pred[i]))

        loss = loss_total / len(data_loader)                    # 计算平均 loss
        acc = metrics.accuracy_score(labels_all, predict_all)   # 计算准确率

        if is_test:
            print('Test: total data %d, wrong prediction %d' %
                  (len(labels_all), wrong_cnt))

        return acc, loss

    def test(self):
        """
        模型测试，使用测试数据集测试模型
        :return:
        """
        acc, loss = self.eval(self.test_loader, is_test=True)
        print('Test: Average Loss: {0:.3f}, Accuracy: {1:>5.2%}'
              .format(loss, acc))
