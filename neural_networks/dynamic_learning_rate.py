#-*-coding:utf-8-*- 
from .back_propagation import back_propagation
from .neural_networks_base import *


if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    df = pd.read_csv('data/iris/iris.data', header = None)
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    print(df)
    training_set = TrainingSet(df, 'class')

    print('输入-训练集:')
    print(training_set.samples)
    print('\n输入-标记集:')
    print(training_set.labels)
    config = {CONST_CONFIG_KEY_TIMES: 20} # 停止条件是训练20轮
    leaning_rate = 0.1 # 学习率
    nn = NeualNetworks([4, 4, 3])
    back_propagation(nn, training_set, leaning_rate, stop_function_by_times, config)