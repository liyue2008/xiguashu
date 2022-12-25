#-*-coding:utf-8-*- 
from .neural_network_base import *
import pandas as pd

def bp_training(training_set: TrainingSet, learning_rate: float, stop_function: Callable[[TrainingSet, SingleHidenLayerNM]], bool) -> SingleHidenLayerNM:
    """逆误差传播算法(Back Progaration)的实现.
    
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
    nm = SingleHidenLayerNM(2, 2, 1)
    
    # 2. repeat
    while True:
        # 3. for all (xk, yk) ∈ trainning_set do
        for index, sample in training_set.samples.iterrows():

            
            # 4. 根据当前参数和式(5.3)计算当前样本的输出^yk
            predict_values = pd.Series(nm.predict(sample.tolist())) # 神经网络预测值
            labeled_values = training_set.labels.iloc[index] # 标记值

            # 5. 根据式(5.10)计算输出层神经元的梯度项gj
            gradient_output_layer_list = []
            for j in range(predict_values.size):
                predict_value = predict_values.iloc[j]
                labeled_value = labeled_values.iloc[j]
                gradient = predict_value * ( 1 - predict_value) * (labeled_value - predict_value)
                gradient_output_layer_list.append(gradient)
            # 6. 根据式(5.15)计算隐层神经元的梯度项eh
           
            # 7. 根据式(5.11)-(5.14)更新连接权whj,vih与阈值θj, γh
        # 8. end for
    # 9. until 达到停止条件
        if stop_function(training_set, nm):
            break