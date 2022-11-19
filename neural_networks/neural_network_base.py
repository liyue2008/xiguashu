#-*-coding:utf-8-*- 
import math
import random
from typing import List, Set
import pandas as pd
from collections.abc import Callable
from decision_tree import DataSet
import numpy as np


def sigmoid_activation_function(x: float) -> float:
    return 1 / (1 + math.exp(-x))
def input_activation_function(x: float) -> float:
    return x


class Neuron:
    """M-P神经元模型. 
    在这个模型中, 神经元接收到来自n个其他神经元传递过来的输入信号, 
    这些输入信号通过带权重的连接(connection)进行传递, 
    神经元接收到的总输入值将与神经元的阀值进行比较, 然后通过"激活函数" (activation function) 处理以产生神经元的输出.

    Attributes
    ----------
    activation_function : Callable
        激活函数.
    threshold : float
        神经元的阈值.
    """
    def __init__(self, activation_function: Callable[[float], float] = sigmoid_activation_function, threshold: float = 0) -> None:
        """创建并初始化神经元.
            
            Parameters
            ----------
            activation_function : Callable
                激活函数, 默认为sigmoid函数.
            threshold : float
                神经元的初始阈值。
                输入初始阈值大于或等于0的时, 所有神经元的初试阈值都设为给定的初始阈值.
                输入初始阈值小于0的时, 所有神经元将给定[0, 1)范围内的随机初始阈值.
        """

        self.activation_function = activation_function
        self.threshold = threshold if threshold >= 0 else random.random()

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
class NMLayer:
    """神经网络的层，包含若干神经元
    
    Attributes
    ----------
    neutons : List[Neuron]
        神经元列表
    """
    def __init__(self, size: int, activation_function: Callable[[float], float] = sigmoid_activation_function, threshold: float = 0) -> None:
        """创建并初始化神经网络中的一层。
            
            Parameters
            ----------
            activation_function : Callable
                激活函数, 默认为sigmoid函数.
            threshold : float
                神经元的初始阈值。
                输入初始阈值大于或等于0的时, 所有神经元的初试阈值都设为给定的初始阈值.
                输入初始阈值小于0的时, 所有神经元将给定[0, 1)范围内的随机初始阈值.
        """
        self.neutons = []
        for _ in range(size):
            self.neutons.append(Neuron(activation_function, threshold))

    def __len__(self) -> int:
         return len(self.neutons)
class NMConnection:
    """神经网络中神经元之间的连接.
    信号传递方向: src --(weight)--> des
    
    Attributes
    ----------
    des : Neuron
        目标神经元.
    src : Neuron
        原神经元.
    weight: float
        连接权重.
    """ 
    def __init__(self, src: Neuron, des: Neuron, weight: float) -> None:
        """初始化神经网络中神经元之间的连接.

        Parameters
        ----------
        src : Neuron
            原神经元.
        des : Neuron
            目标神经元.
        weight : float
            连接权重.
            输入大于或等于0的时, 连接权重设为给定的输入值.
            输入小于0的时, 连接权重设为[0, 1)范围内的随机值.
        """
        self.src = src
        self.des = des
        self.weight = weight if weight >= 0 else random.random()
class NeualNetworks:
    """多层前馈神经网络.

    Attributes
    ----------
    layers: List[NMLayer]
        层的列表, 每层包含若干神经元.
    connections: Set[NMConnection]
        网络中神经元的连接集合.
    """
    def __init__(self, layers: List[NMLayer], connections: Set[NMConnection]) -> None:
        self.layers = layers
        self.connections = connections
    def predict(self, input: List[float]) -> List[float]
class SingleHidenLayerNM(NeualNetworks):
    """单隐层前馈神经网络."""



    def __init__(self, input_layer_size: int, hiden_layer_size: int, output_layer_size: int) -> None:
        """初始化单隐层前馈神经网络.
        激活函数为Sigmoid函数, 所有连接权值和输出阈值为(0, 1)范围内的随机值.
        Parameters
        ----------
        input_layer_size : int
            单隐层前馈神经网络中输入层神经元个数.
        hiden_layer_size : int
            单隐层前馈神经网络中隐层神经元个数.
        output_layer_size : int
            单隐层前馈神经网络中输出层神经元个数.

        """
        
        # 初始化输入层
        input_layer = NMLayer(input_layer_size, input_activation_function)
        # 初始化隐层
        hiden_layer = NMLayer(hiden_layer_size)
        # 初始化输出层
        output_layer = NMLayer(output_layer_size, threshold = -1)

        connections = {}
        # 初始化输入层-隐层连接
        for src in input_layer.neutons:
            for dest in hiden_layer.neutons:
                connections.add(NMConnection(src, dest, -1))
        
        for src in hiden_layer.neutons:
            for dest in output_layer.neutons:
                connections.add(NMConnection(src, dest, -1))

        super().__init__([input_layer, hiden_layer, output_layer], connections)
