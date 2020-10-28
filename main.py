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

import sys

from recognizer import Recognizer


def main():
    data_path = {
        'data_dir': 'data/',
        'train_image': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_image': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz',
    }
    dropout_prob = 0.2
    halve_conv_kernels = False

    # 初始化手写数字识别器
    recognizer = Recognizer(data_path=data_path,
                            dropout_prob=dropout_prob,
                            halve_conv_kernels=halve_conv_kernels)
    # 准备数据集
    recognizer.prepare()
    # 模型训练
    recognizer.train()
    # 模型测试
    recognizer.test()


if __name__ == '__main__':
    sys.exit(main())
