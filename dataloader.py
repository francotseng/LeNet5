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
import gzip
import struct
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader


def load_image_labels(data_dir, image_file, label_file):
    """
    加载一个数据集
    :param data_dir: 数据集目录
    :param image_file: 图像文件名
    :param label_file: 标签文件名
    :return: 图像数据 nd-array 和标签数据 nd-array
    """
    image_path = os.path.join(data_dir, image_file)
    label_path = os.path.join(data_dir, label_file)
    with gzip.open(label_path) as f_label:
        magic, num = struct.unpack('>II', f_label.read(8))
        label = np.fromstring(f_label.read(), dtype=np.int8)
    with gzip.open(image_path, 'rb') as f_image:
        magic, num, rows, cols = struct.unpack('>IIII', f_image.read(16))
        image = np.fromstring(f_image.read(), dtype=np.uint8).\
            reshape((len(label), rows, cols))
    return image, label


def load_data(data_dir, image_file, label_file, batch_size=128):
    """
    加载数据集并返回 DataLoader
    :param data_dir: 数据集目录
    :param image_file: 图像文件名
    :param label_file: 标签文件名
    :param batch_size: 每次读取数据的个数
    :return: 数据集的 DataLoader
    """
    train_images, train_labels = \
        load_image_labels(data_dir, image_file, label_file)

    # print(train_images.shape, train_labels.shape)

    train_dataset = TensorDataset(torch.tensor(train_images).float(),
                                  torch.tensor(train_labels).long())
    train_loader = DataLoader(dataset=train_dataset, shuffle=True,
                              batch_size=batch_size)
    return train_loader


def show_image(image, label):
    """
    绘制图像
    :param image: 图像数据
    :param label: 在图像上显示的标签
    :return:
    """
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.show()
