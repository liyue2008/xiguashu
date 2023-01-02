#-*-coding:utf-8-*- 
from .neural_networks_base import *

def accumulated_back_propagation(nn: NeualNetworks, training_set: TrainingSet, learning_rate: float, stop_function: Callable[[TrainingSet, NeualNetworks, Any, Dict[str, Any]], Tuple[bool, Any]], config: Dict[str, Any] = {}) -> None:
    """累积误差传播算法(Accumulated Back Propagation)的实现.
    
    Parameters
    ----------
    nm: NeualNetworks
        被训练的神经网络.
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
        ce = nn.cumulative_error(training_set)
        print('已训练%d轮, 累积误差: %.4f' % (times, ce))
        times = times + 1
        print('当前网络:')
        print(nn)
        
        # 3. for all (xk, yk) ∈ trainning_set do
        for index, sample in training_set.samples.iterrows():

            # 4. 根据当前参数和式(5.3)计算当前样本的输出^yk
            predict_matrix = nn.predict(sample.tolist())
            predict_values = pd.Series(predict_matrix[len(predict_matrix) - 1]) # 神经网络预测值
            labeled_values = training_set.labels.loc[index] # 标记值

            # 5. 根据式(5.10)计算输出层神经元的梯度项gj
            gradient_output_layer = []
            for j in range(predict_values.size):
                predict_value = predict_values.iloc[j]
                labeled_value = labeled_values.iloc[j]
                gradient = predict_value * (1 - predict_value) * (labeled_value - predict_value)
                gradient_output_layer.append(gradient)
            
            gradient_upper_layer = gradient_output_layer # 上一层神经元的梯度
            for layer_index in reversed(range(len(nn.layers) - 1)): # 自顶向下遍历所有的隐层和输入层，计算每一隐层神经元的梯度项
                output_current_layer = predict_matrix[layer_index] # 本层神经元输出
                # 6. 根据式(5.15)计算隐层神经元的梯度项eh
                gradient_current_layer = calc_layer_gradient(nn, layer_index, output_current_layer, gradient_upper_layer) # 本层神经元梯度
                
                # 7. 根据式(5.11)-(5.14)更新连接权whj,vih与阈值θj, γh.
                # 这里只更新whj和θj. 
                # 原因是, vih与γh本质上更新公式是一样的, 就是循环到下一层的whj和θj.

                # 7.1 根据式(5.11)更新连接权whj
                for h, n in enumerate(nn.layers[layer_index].neutons): # 遍历当前隐层每个神经元
                    for j, _ in enumerate(nn.layers[layer_index + 1].neutons):  # 遍历上层神经元
                        gj = gradient_upper_layer[j]
                        delta_whj = learning_rate * gj * output_current_layer[h]
                        n.output_connections[j].delta_weights.append(delta_whj)

                # 7.2 根据式(5.12)更新阈值θj
                for j, nu in enumerate(nn.layers[layer_index + 1].neutons): 
                    gj = gradient_upper_layer[j]
                    delta_theta_j = -1 * learning_rate * gj
                    nu.delta_thresholds.append(delta_theta_j)

                gradient_upper_layer = gradient_current_layer.copy()
        # 8. end for

        # 更新连接权值和神经元阈值
        for layer_index, layer in enumerate(nn.layers):
            if layer_index == 0: # 忽略输入层
                continue
            for n in layer.neutons:
                # 更新神经元阈值
                n.threshold += sum(n.delta_thresholds) / len(n.delta_thresholds)
                n.delta_thresholds = []
                # 更新神经元输入连接的阈值
                for ic in n.input_connections:
                    ic.weight += sum(ic.delta_weights) / len(ic.delta_weights)
                    ic.delta_weights = []
                

        # 9. until 达到停止条件
        (stop, acc) = stop_function(training_set, nn, acc, config)
        if stop:
            break
    ce = nn.cumulative_error(training_set)
    print('已训练%d轮, 累积误差: %.4f' % (times, ce))
    times = times + 1
    print('当前网络:')
    print(nn)

if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
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
    config = {CONST_CONFIG_KEY_TIMES: 20} # 停止条件是训练20轮
    leaning_rate = 0.1 # 学习率
    nn = NeualNetworks([2, 2, 1])
    accumulated_back_propagation(nn, training_set, leaning_rate, stop_function_by_times, config)