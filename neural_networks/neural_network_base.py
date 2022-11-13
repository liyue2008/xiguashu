#-*-coding:utf-8-*- 
import math
import pandas as pd
from collections.abc import Callable
from decision_tree import DataSet
import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

class Neuron:
    """M-P神经元模型. 
    在这个模型中, 神经元接收到来自n个其他神经元传递过来的输入信号, 
    这些输入信号通过带权重的连接(connection)进行传递, 
    神经元接收到的总输入值将与神经元的阀值进行比较, 然后通过"激活函数" (activation function) 处理以产生神经元的输出.

    Attributes
    ----------
    activation_function : Callable
        隐层激活函数, 默认为sigmoid函数.
    threhold : float
        神经元的阈值, 默认为0.

    """
    def __init__(self, activation_function: Callable[[float], float] = sigmoid, threhold: float = 0) -> None:
        self.activation_function = activation_function
        self.threhold = threhold

    def active(self, inputs: np.ndarray, connection_weights: np.ndarray) -> float:
        """激活神经元, 计算输出值.

        Parameters
        ----------
        inputs: ndarray
            上游神经元的输入值数组.
        connection_weights: ndarray
            神经元与上游神经元的链接权重数组.

        Returns
        ----------
        经过激活函数计算后的输出值, 若输出值大于0, 则表示神经元被激活.
        """
        return np.sum(inputs * connection_weights) - self.threhold

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
