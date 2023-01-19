#-*-coding:utf-8-*- 
from .back_propagation import back_propagation
from .neural_networks_base import *
CONST_CONFIG_KEY_LEARNING_RATE_EPOCH = 'learning_rate_epoch' # 每训练多少轮次调整学习率
CONST_CONFIG_KEY_LEARNING_RATE_GAMMA = 'learning_rate_gamma' # 学习率调整系数

class DynamicLRAccumulator:
    def __init__(self, learning_rate: float, epoch: int, gamma: float) -> None:
        assert (learning_rate and type(learning_rate) is float and learning_rate > 0 and learning_rate < 1), "初始学习率learning_rate必须是(0, 1)范围的浮点数!"
        assert (epoch and type(epoch) is int and epoch > 0), "epoch必须是大于0的整数!"
        assert (gamma and type(gamma) is float and gamma > 0 and gamma < 1), "gamma必须是(0, 1)范围的浮点数!"

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.gamma = gamma
        self.current_epoch = 0

def dynamic_learning_rate_provider(accumulator: DynamicLRAccumulator, config: Dict[str, Any]) -> Tuple[float, DynamicLRAccumulator]:
    """最简单的动态调整学习率算法, 固定每n次训练, 将学习率乘以一个[0, 1)的系数, 以降低学习率.
    """
    if not accumulator:
        accumulator = DynamicLRAccumulator(config[CONST_CONFIG_KEY_LEARNING_RATE], config[CONST_CONFIG_KEY_LEARNING_RATE_EPOCH], config[CONST_CONFIG_KEY_LEARNING_RATE_GAMMA])
    else:
        if accumulator.current_epoch != 0 and accumulator.current_epoch % accumulator.epoch == 0:
            accumulator.learning_rate = accumulator.learning_rate * accumulator.gamma
    accumulator.current_epoch = accumulator.current_epoch + 1
    return (accumulator.learning_rate, accumulator)

def bp_nn(training_set_path: str, learning_rate_provider: Callable[[Any, Dict[str, Any]], Tuple[float, Any]], stop_function: Callable[[TrainingSet, NeualNetworks, Any, Dict[str, Any]], Tuple[bool, Any]], config: Dict[str, Any] = {}) -> None:
    
    df = pd.read_csv(training_set_path, header = None)
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    training_set = TrainingSet(df, 'class')

    print('输入 - 训练集:')
    print(training_set)
    print('输入 - 配置: %s' % config)
    nn = NeualNetworks([4, 4, 3])
    back_propagation(nn, training_set, learning_rate_provider, stop_function, config)
    print('输出-神经网络:')
    print(nn)

if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    config = {
        # CONST_CONFIG_KEY_VERBOSE: True, # 打印详细训练信息
        CONST_CONFIG_KEY_CE: 0.02, # 停止条件是累积误差小于0.05
        CONST_CONFIG_KEY_STOP_EPOCH: 1000, # 最多训练1000轮 
        CONST_CONFIG_KEY_LEARNING_RATE: 0.2, # 起始学习率
        CONST_CONFIG_KEY_LEARNING_RATE_EPOCH: 10, # 每多少轮训练调整一次学习率
        CONST_CONFIG_KEY_LEARNING_RATE_GAMMA: 0.93 # 每次调整学习率的系数: 新学习率 = 旧学习率 * 系数
    } 
    print('训练集: iris, 固定学习率.')
    bp_nn('data/iris/iris.data', fixed_learning_rate_provider, stop_function_by_target_ce, config)

    print('训练集: iris, 动态学习率.')
    bp_nn('data/iris/iris.data', dynamic_learning_rate_provider, stop_function_by_target_ce, config)

    print('训练集: bezdekIris, 固定学习率.')
    bp_nn('data/iris/bezdekIris.data', fixed_learning_rate_provider, stop_function_by_target_ce, config)

    print('训练集: bezdekIris, 动态学习率.')
    bp_nn('data/iris/bezdekIris.data', dynamic_learning_rate_provider, stop_function_by_target_ce, config)