#-*-coding:utf-8-*- 
from __future__ import annotations
import math
import random
from typing import List
from collections.abc import Callable
import numpy as np
import pandas as pd


def sigmoid_activation_function(x: float) -> float:
    return 1 / (1 + math.exp(-x))
def input_activation_function(x: float) -> float:
    return x

class TrainingSet:
    """包含样本和标记的训练集.
    输入样本x由d个属性描述, 即d维实值向量.
    输出样本(即标记值)y由l个属性描述, 即l维实值向量.
    
    Attributes
    ----------
    samples: pd.DataFrame
        输入样本x由d个属性描述, 即d维实值向量.
    labels: pd.DataFrame
        输出样本(即标记值)y由l个属性描述, 即l维实值向量.
    """
    def __init__(self, samples: pd.DataFrame, labels: pd.DataFrame) -> None:
        self.samples = samples
        self.labels = labels
        assert (len(samples.index) == len(labels.index)), "Samples and labels should be same size!"
    
    def __len__(self) -> int:
        return len(self.samples.index)

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
        return self.__calc_layer_output(input, len(self.layers) - 1)

    def __calc_layer_output(self, input: List[float], layer_index: int) -> List[float]:
        """对于给定的输入input, 计算神经网络从输入层开始, 到指定层layer_index的输出值.
        """
        # 检查input中元素的个数应等于输入层神经元个数
        self.__predict_input_check(input)
        assert (layer_index >=0 and layer_index < len(self.layers)), F"layer_index 的取值范围应是：[0, {len(self.layers)}), 实际值: {layer_index}"
        layer_output = []
        for index in range(layer_index + 1):
            layer = self.layers[index]
            pre_layer_output = layer_output
            layer_output = []
            for i, n in enumerate(layer.neutons):
                if not pre_layer_output: # 输入层
                    layer_output.append(n.input(input[i]))
                else: # 非输入层
                    layer_output.append(n.active(pre_layer_output))
        return layer_output

    def back_propagation(self, training_set: TrainingSet, learning_rate: float, stop_function: Callable[[TrainingSet, SingleHidenLayerNM]], bool) -> None:
        """逆误差传播算法(Back Propagation)的实现.
        
        Parameters
        ----------
        training_set : TrainingSet
            训练集, 包含训练数据集和标记集.
        learning_rate: float
            学习率, 取值范围(0, 1), 学习率控制着算法每一轮迭代中的更新步长.

        Returns
        -------
        连接权与阈值确定的多层前馈神经网络.
        """
       
        # 1. 在(0, 1)范围内随机初始化网络中的所有连接权和阈值 
        # 2. repeat
        while True:
            # 3. for all (xk, yk) ∈ trainning_set do
            for index, sample in training_set.samples.iterrows():

                # 4. 根据当前参数和式(5.3)计算当前样本的输出^yk
                predict_values = pd.Series(self.predict(sample.tolist())) # 神经网络预测值
                labeled_values = training_set.labels.iloc[index] # 标记值

                # 5. 根据式(5.10)计算输出层神经元的梯度项gj
                gradient_output_layer = []
                for j in range(predict_values.size):
                    predict_value = predict_values.iloc[j]
                    labeled_value = labeled_values.iloc[j]
                    gradient = predict_value * ( 1 - predict_value) * (labeled_value - predict_value)
                    gradient_output_layer.append(gradient)
               
                gradient_upper_layer = gradient_output_layer # 上一层神经元的梯度
                for layer_index in reversed(range(0, self.layers - 1)): # 自顶向下遍历所有的隐层和输入层，计算每一隐层神经元的梯度项
                    output_current_layer = self.__calc_layer_output(sample, layer_index) # 本层神经元输出
                    # 6. 根据式(5.15)计算隐层神经元的梯度项eh
                    gradient_current_layer = self.__calc_layer_gradient(layer_index, output_current_layer, gradient_upper_layer) # 本层神经元梯度
                    
                    # 7. 根据式(5.11)-(5.14)更新连接权whj,vih与阈值θj, γh.
                    # 这里只更新whj和θj. 
                    # 原因是, vih与γh本质上更新公式是一样的, 就是循环到下一层的whj和θj.

                    # 7.1 根据式(5.11)更新连接权whj
                    for h, n in enumerate(self.layers[layer_index].neutons): # 遍历当前隐层每个神经元
                       for j, _ in enumerate(self.layers[layer_index + 1].neutons):  # 遍历上层神经元
                            gj = gradient_upper_layer[j]
                            delta_whj = learning_rate * gj * output_current_layer[h]
                            n.output_connections[j].weight += delta_whj

                    # 7.2 根据式(5.12)更新阈值θj
                    for j, nu in enumerate(self.layers[layer_index + 1].neutons): 
                        gj = gradient_upper_layer[j]
                        delta_theta_j = -1 * learning_rate * gj
                        nu.threshold += delta_theta_j

                    gradient_upper_layer = gradient_current_layer.copy()
            # 8. end for
            # 9. until 达到停止条件
            if stop_function(training_set, self):
                break
    def __calc_layer_gradient(self,layer_index: int, output_current_layer: List[float], gradient_upper_layer: List[float]) -> List[float]:
        """根据式(5.15)计算一层神经元的梯度项.
        
        Parameters
        ----------
        layer_index : int
            计算第layer_index层.
        output_current_layer : List[float]
            当前层神经元的输出值.       
        gradient_upper_layer: List[float]
            当前层上一层神经元的梯度.
        
        Returns
        -------
        true: D中样本在A上取值相同。
        false: D中样本在A上取值不完全相同。
        """
        layer_gradient = [] # 本层神经元梯度
        for h, n in enumerate(self.layers[layer_index].neutons): # 遍历当前隐层每个神经元
        # 计算与上一层连接权与梯度的乘积之和
            total_gj_x_whj = 0
            for j, gj in enumerate(gradient_upper_layer): 
                whj = n.output_connections[j].weight
                total_gj_x_whj += gj * whj
            # 计算当前神经元的输出值
            bh = output_current_layer[h] 
            # 计算当前神经元的梯度
            eh = bh * (1 - bh) * total_gj_x_whj
            layer_gradient.append(eh)
        return layer_gradient


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
