#-*-coding:utf-8-*- 
import math
from collections.abc import Callable

import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

class SingleHidenLayerFeedforwardNeualNetworks:
    """单隐层前馈神经网络.
    默认所有连接权值和输出阈值为(0, 1)范围内的随机值.

    Attributes
    ----------
    input_layer_size : int
        单隐层前馈神经网络中输入层神经元个数.
    hiden_layer_size : int
        单隐层前馈神经网络中隐层神经元个数.
    output_layer_size : int
        单隐层前馈神经网络中输出层神经元个数.
    activation_function : Callable
        隐层激活函数, 默认为sigmoid函数.
    input_hiden_weights : ndarray
        输入层与隐层连接的权值, 二维numpy数组. 
        例如: 输入层第i个神经元与隐层第j个神经元的权值为: input_hiden_weights[i, j].
    hiden_output_weights : ndarray
        隐层与输出层连接的权值, 二维numpy数组. 
        例如: 隐层第i个神经元与输出层第j个神经元的权值为: hiden_output_weights[i, j].
    thresholds : ndarray
        输出层权值, 一维数组.
        例如: 输出层第i个神经元的阈值为: thresholds[i].
    """



    def __init__(self, input_layer_size: int, hiden_layer_size: int, output_layer_size: int, activation_function: Callable[[float], float] = sigmoid) -> None:
        self.input_layer_size = input_layer_size
        self.hiden_layer_size = hiden_layer_size
        self.output_layer_size = output_layer_size
        self.activation_function = activation_function
        self.input_hiden_weights = np.random.rand(input_layer_size, hiden_layer_size)
        self.hiden_output_weights = np.random.rand(hiden_layer_size, output_layer_size)
        self.thresholds = np.random.rand(output_layer_size)
