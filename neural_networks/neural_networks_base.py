#-*-coding:utf-8-*- 
from __future__ import annotations
from ast import Tuple
import math
import random
from typing import Any, Dict, List
from collections.abc import Callable
import numpy as np
import pandas as pd


def sigmoid_activation_function(x: float) -> float:
    return 1 / (1 + math.exp(-x))
def input_activation_function(x: float) -> float:
    return x

CONST_CONFIG_KEY_TIMES = 'times'
def stop_function_by_times(training_set: TrainingSet, neualNetworks: NeualNetworks, accumulator: int, config: Dict[str, Any]) -> Tuple[bool, int]:
    """ 判断训练停止条件的函数, 达到既定训练次数后就停止. 
        Parameters
        ----------
            training_set: TrainingSet
                训练集, 包含训练数据集和标记集.
            neualNetworks: NeualNetworks
                被训练的神经网络.
            accumulator: Any
                累加器. 这是一个任意类型的变量, 其类型可由实现自行定义, 用于多次调用停止函数时记录一些需要累计的数据.
            config: Dict[str, Any]
                训练配置, 这里需接收一个停止次数参数, key为CONST_CONFIG_KEY_TIMES, value必须为正整数.
        Returns
        -------
            stop: bool
                True: 停止训练, False: 继续训练.
            accumulator: Any
                累加器. 
    """
    training_times = config[CONST_CONFIG_KEY_TIMES]
    assert (training_times and type(training_times) is int and training_times > 0), "停止次数(参数config[%s])必须是正整数!" % (CONST_CONFIG_KEY_TIMES)
    if not accumulator:
        accumulator = 0
    accumulator = accumulator + 1
    return (accumulator >= training_times, accumulator)

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
    activation_function : Callable[[float], float]
        激活函数.
    threshold : float
        神经元的阈值.
    input_connections: List[NMConnection]
        输入连接.
    output_connections: List[NMConnection]
        输出连接.
    """
    def __init__(self, activation_function: Callable[[float], float] = sigmoid_activation_function, threshold: float = 0, random_threshold: bool = True) -> None:
        """创建并初始化神经元.
            
            Parameters
            ----------
            activation_function: Callable[[float], float] = sigmoid_activation_function
                激活函数, 默认为sigmoid函数.
            threshold: float = 0
                如果random_threshold == False, 将除输入层以外的所有神经元阈值初始化为固定值threshold.
            random_threshold: bool = True
                忽略给定的阈值threshold, 将除输入层以外的所有神经元阈值初始化为(0, 1)范围内的随机值.
        """

        self.activation_function = activation_function
        self.threshold = self.__random_threshold() if random_threshold else threshold
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
        return self.activation_function(input - self.threshold)

    def __random_threshold(self) -> float:
        """将阈值设置为(0, 1)随机值."""
        self.threshold = random.random()
        return self.threshold if self.threshold != 0 else self.__random_threshold()

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

        weights = list(map(lambda c: c.weight, self.__input_connections))
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
    def __init__(self, size: int, activation_function: Callable[[float], float] = sigmoid_activation_function, threshold: float = 0, random_threshold: bool = True) -> None:
        """创建并初始化神经网络中的一层。
            
            Parameters
            ----------
            activation_function : Callable
                激活函数, 默认为sigmoid函数.
            threshold: float = 0
                如果random_threshold == False, 将除输入层以外的所有神经元阈值初始化为固定值threshold.
            random_threshold: bool = True
                忽略给定的阈值threshold, 将除输入层以外的所有神经元阈值初始化为(0, 1)范围内的随机值.
        """
        self.neutons: List[Neuron] = []
        for _ in range(size):
            self.neutons.append(Neuron(activation_function, threshold, random_threshold))

    def __len__(self) -> int:
         return len(self.neutons)
    @property
    def thresholds(self):
        """每个神经元的阈值数组."""
        return np.array(list(map(lambda n: n.threshold, self.neutons)))
    @property
    def input_weigths(self) -> np.matrix[float, float]:
        """本层与下层连接权值. 返回值是一个二维矩阵, 列数=本层神经元个数，行数=下层神经元的个数.
        """
        weight_2d_list = list(map(lambda n: list(map(lambda c: c.weight, n.input_connections)), self.neutons))
        return np.matrix(weight_2d_list).T
    def __str__(self) -> str:
        np.set_printoptions(precision=3)
        ret = '%s\n' % self.thresholds

        weight_matrix = self.input_weigths
        if weight_matrix.size > 0:
            ret += '%s\n' % weight_matrix
        return ret
    

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
    def __init__(self, src: Neuron, des: Neuron, weight: float = 0, random_weight: bool= True) -> None:
        """初始化神经网络中神经元之间的连接.

        Parameters
        ----------
        src : Neuron
            原神经元.
        des : Neuron
            目标神经元.
        weight: float = 0
            如果random_weight == False, 将所有连接权值设置为固定值weight.
        random_weight: bool = True
            忽略给定的连接权值weight, 将所有连接权值设置为(0, 1)范围内的随机值.
        """
        self.src: Neuron = src
        self.des: Neuron = des
        self.weight: float =  self.__random_weight() if random_weight else weight
        src.append_connection(self)
        des.append_connection(self)
    def __random_weight(self) -> float:
        """将连接权值设置为(0, 1)随机值."""
        self.weight = random.random()
        return self.weight if self.weight != 0 else self.random_weight()
class NeualNetworks:
    """多层前馈神经网络.

    Attributes
    ----------
    layers: List[NMLayer]
        层的列表, 每层包含若干神经元.
    """
    def __init__(self, neuron_count: List[int], threshold: float = 0, random_threshold: bool = True, weight: float = 0, random_weight: bool = True, activation_function: Callable[[float], float] = sigmoid_activation_function) -> None:
        """初始化多层前馈神经网络.
        
        Parameters
        ----------
        neuron_count: List[int]
            每层神经元的个数.
            len(neuron_count)就是神经网络的层数.
            第0层为输入层, 第len(neuron_count) - 1层为输出层, 其它层为隐层.
            neuron_count[i]为第i层神经元的个数.
        threshold: float = 0
            如果random_threshold == False, 将除输入层以外的所有神经元阈值初始化为固定值threshold.
        random_threshold: bool = True
            忽略给定的阈值threshold, 将除输入层以外的所有神经元阈值初始化为(0, 1)范围内的随机值.
        weight: float = 0
            如果random_weight == False, 将所有连接权值设置为固定值weight.
        random_weight: bool = True
            忽略给定的连接权值weight, 将所有连接权值设置为(0, 1)范围内的随机值.
        activation_function: Callable[[float], float] = sigmoid_activation_function
            隐层和输出层的激活函数, 默认为sigmoid函数.
        """
        # 初始化每层的神经元
        self.layers = []
        for index, size in enumerate(neuron_count):
            if index == 0: # 输入层
                self.layers.append(NMLayer(size, input_activation_function, 0, False))
            else: # 隐层和输出层
                self.layers.append(NMLayer(size, activation_function, threshold, random_threshold))
        # 初始化相邻层的神经元之间的连接
        self.__init_layer_connections(weight, random_weight)
    def __str__(self) -> str:
        layers = list(map(lambda layer:str(layer), self.layers))
        ret = ''
        for i, layer_str in reversed(list(enumerate(layers))):
            ret += '第%d层%d个神经元(阈值, 连接权值):\n' % (i, len(self.layers[i]))
            ret += layer_str
        return ret   
    def __init_layer_connections(self, weight: float = 0, random_weight: bool = True) -> None:
        """初始化神经网络的连接.
        
        Parameters
        ----------
        weight: float = 0
            如果random_weight == False, 将所有连接权值设置为固定值weight.
        random_weight: bool = True
            忽略给定的连接权值weight, 将所有连接权值设置为(0, 1)范围内的随机值.
        """
        for layer_index in range(len(self.layers)):
            if layer_index + 1 >= len(self.layers):
                break
            layer = self.layers[layer_index]
            upper_layer = self.layers[layer_index + 1]
            for src in layer.neutons:
                for dest in upper_layer.neutons:
                    NMConnection(src, dest, weight, random_weight)

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

    def back_propagation(self, training_set: TrainingSet, learning_rate: float, stop_function: Callable[[TrainingSet, NeualNetworks, Any, Dict[str, Any]], Tuple[bool, Any]], config: Dict[str, Any] = {}) -> None:
        """逆误差传播算法(Back Propagation)的实现.
        
        Parameters
        ----------
        training_set: TrainingSet
            训练集, 包含训练数据集和标记集.
        learning_rate: float
            学习率, 取值范围(0, 1), 学习率控制着算法每一轮迭代中的更新步长.
        stop_function: Callable[[TrainingSet, NeualNetworks, Any], bool])
            判断训练停止条件的函数. 
            Parameters
            ----------
                training_set: TrainingSet
                    训练集, 包含训练数据集和标记集.
                neualNetworks: NeualNetworks
                    被训练的神经网络.
                accumulator: Any
                    累加器. 这是一个任意类型的变量, 其类型可由实现自行定义, 用于多次调用停止函数时记录一些需要累计的数据.
                config: Dict[str, Any]
                    训练配置, 可定义.
            Returns
            -------
                stop: bool
                    True: 停止训练, False: 继续训练.
                accumulator: Any
                    累加器.
        config: Dict[str, Any]
            训练配置, 可定义.
        Returns
        -------
        连接权与阈值确定的多层前馈神经网络.
        """
       
        # 1. 在(0, 1)范围内随机初始化网络中的所有连接权和阈值 
        # 2. repeat
        acc = None
        times = 0
        while True:
            # 打印误差和网络
            ce = self.cumulative_error(training_set)
            print('已训练%d轮, 累积误差: %.4f' % (times, ce))
            times = times + 1
            print('当前网络:')
            print(self)
            # 3. for all (xk, yk) ∈ trainning_set do
            for index, sample in training_set.samples.iterrows():

                # 4. 根据当前参数和式(5.3)计算当前样本的输出^yk
                predict_values = pd.Series(self.predict(sample.tolist())) # 神经网络预测值
                labeled_values = training_set.labels.loc[index] # 标记值

                # 5. 根据式(5.10)计算输出层神经元的梯度项gj
                gradient_output_layer = []
                for j in range(predict_values.size):
                    predict_value = predict_values.iloc[j]
                    labeled_value = labeled_values.iloc[j]
                    gradient = predict_value * (1 - predict_value) * (labeled_value - predict_value)
                    gradient_output_layer.append(gradient)
               
                gradient_upper_layer = gradient_output_layer # 上一层神经元的梯度
                for layer_index in reversed(range(len(self.layers) - 1)): # 自顶向下遍历所有的隐层和输入层，计算每一隐层神经元的梯度项
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
            (stop, acc) = stop_function(training_set, self, acc, config)
            if stop:
                break
        ce = self.cumulative_error(training_set)
        print('已训练%d轮, 累积误差: %.4f' % (times, ce))
        times = times + 1
        print('当前网络:')
        print(self)
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
    def mean_squared_error(self, sample: List[float], lable: List[float]) -> float:
        """计算神经网络在给定样本sample相对于标记label的均方误差(MSE).
        
        Parameters
        ----------
        sample: List[float]
            样本数据, 数组长度应等于网络输入层神经元个数.
        lable: List[float]
            标记数据, 数组长度应等于网络输出层神经元个数.
        
        Returns
        -------
            给定样本sample相对于标记label的均方误差(MSE).
        """
        predicts = self.predict(sample)
        mse = 0
        for j, predict in enumerate(predicts):
            mse += ((predict - lable[j]) ** 2) / 2
        return mse
    def cumulative_error(self, training_set: TrainingSet) -> float:
        """计算神经网络在给定训练集training_set上的累积误差.
        
        Parameters
        ----------
        training_set: TrainingSet
            给定训练集, 包含样本和标记
        Returns
        -------
            神经网络在给定训练集training_set上的累积误差.
        """
        ce = 0
        for k, sample in training_set.samples.iterrows():
            label = training_set.labels.loc[[k]].values.flatten()
            ce += self.mean_squared_error(sample.tolist(), label.tolist())
        return ce / len(training_set)

if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    config = {CONST_CONFIG_KEY_TIMES: 2000}
    df = pd.read_csv('data/西瓜数据集 3.0.csv')
    df.set_index('编号', inplace=True)
    sample_set = df[['密度', '含糖率']]
    label_set = df[['好瓜']]
    label_set.loc[label_set['好瓜'] == '是', '好瓜'] = 1
    label_set.loc[label_set['好瓜'] == '否', '好瓜'] = 0
    print('输入-训练集:')
    print(sample_set)
    print('\n输入-标记集:')
    print(label_set)
    training_set = TrainingSet(sample_set, label_set)
    leaning_rate = 0.1
    nm = NeualNetworks([2, 2, 1])
    nm.back_propagation(training_set, leaning_rate, stop_function_by_times, config)
