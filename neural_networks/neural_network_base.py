#-*-coding:utf-8-*- 
from __future__ import annotations
import math
import random
from typing import List
from collections.abc import Callable
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
        self.__input_connections: List[NMConnection] = []
        self.__output_connections: List[NMConnection] = []
    @property
    def input_connections(self) -> List[NMConnection]:
        return self.__input_connections.copy()
    @property
    def output_connections(self) -> List[NMConnection]:
        return self.__output_connections.copy()
    def input(self, input: float) -> float:
        """激活输入层神经元, 计算输出值.

        Parameters
        ----------
        input: float
            神经元的输入值

        Returns
        ----------
        经过激活函数计算后的输出值, 若输出值大于0, 则表示神经元被激活.
        """
        return self.activation_function(input - self.threshold) if input > self.threshold else 0
    def active(self, input: List[float]) -> float:
        """激活神经元, 计算输出值.

        Parameters
        ----------
        input: List[float]
            上游神经元的输入值数组.

        Returns
        ----------
        经过激活函数计算后的输出值, 若输出值大于0, 则表示神经元被激活.
        """

        weights = []
        for c in self.__input_connections:
            weights.append(c.weight)
        np_weights = np.array(weights)
        np_input = np.array(input)
        return self.input(np.sum(np_input * np_weights))


    def append_connection(self, connection: NMConnection) -> None:
        if connection.src == self:
            self.__output_connections.append(connection)
        elif connection.des == self:
            self.__input_connections.append(connection)
        else:
            raise ValueError('只能添加与神经元有连接关系的connection!')
class NMLayer:
    """神经网络的层, 包含若干神经元
    
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
        self.neutons: List[Neuron] = []
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
        self.src: Neuron = src
        self.des: Neuron = des
        self.weight: float = weight if weight >= 0 else random.random()
        src.append_connection(self)
        des.append_connection(self)
class NeualNetworks:
    """多层前馈神经网络.

    Attributes
    ----------
    layers: List[NMLayer]
        层的列表, 每层包含若干神经元.
    """
    def __init__(self, layers: List[NMLayer]) -> None:
        self.layers = layers
  
    def __predict_input_check(self, input: List[float]) -> None:
        """检查input中元素的个数应等于输入层神经元个数"""
        li = len(input)
        ln = len(self.layers[0].neutons)
        if li != ln:
            raise ValueError('input中元素的个数(%d)应等于输入层神经元个数(%d)!' % (li, ln))

        
    def predict(self, input: List[float]) -> List[float]:
        """计算神经网络的输出"""
        # 检查input中元素的个数应等于输入层神经元个数
        self.__predict_input_check(input)
        layer_output = []
        for layer in self.layers:
            pre_layer_output = layer_output
            layer_output = []
            for i, n in enumerate(layer.neutons):
                if not pre_layer_output: # 输入层
                    layer_output.append(n.input(input[i]))
                else: # 非输入层
                    layer_output.append(n.active(pre_layer_output))
        return layer_output



    
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

        # 初始化输入层-隐层连接
        for src in input_layer.neutons:
            for dest in hiden_layer.neutons:
                NMConnection(src, dest, -1)
        
        for src in hiden_layer.neutons:
            for dest in output_layer.neutons:
                NMConnection(src, dest, -1)

        super().__init__([input_layer, hiden_layer, output_layer])
