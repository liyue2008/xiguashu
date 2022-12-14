# 第五章 神经网络

这一章的习题都放在[neural_networks](../neural_networks/)目录下, 目录结构如下:

```bash
neural_networks
├── __init__.py
├── accumulated_back_propagation.py  # 累积BP算法实现
├── back_propagation.py # 标准BP算法实现
└── neural_networks_base.py # 神经网络数据模型、预测算法、均方差和累积误差的实现
```

## 5.1 试述线性函数$f(x) = w^Tx$用作神经元激活函数的缺陷.

答: 线性函数的线下叠加仍然是线性函数, 而理想的激活函数应具有跃迁或者阶跃的特性.

## 5.2 试述使用 图5.2(b) 的激活函数的神经元与对率回归的联系.

答: 对数几率回归(Logistic Regression), 简称对率回归. 用来将空间上的向量映射为0或1, 也就是用于做二分类. 单位阶跃函数是最理想的一种, 但因为不连续, 所以不是单调可微函数. 而$sigmoid$函数是单位阶跃函数的理想替代, 在一定程度上近似单位阶跃函数, 并且单调可微, 是任意阶可导的凸函数.  使用 $Sigmoid$ 激活函数, 每个神经元几乎和对率回归相同, 只不过对率回归在$sigmoid(x)>0.5$时输出为1, 而神经元直接输出$sigmoid(x)$。

## 5.3 对于图5.7中的$v_{ih}$, 试推导出BP算法中的更新公式(5.13).

这题暂且摆烂了, 真的不是很擅长公式推导.  
解题的思路是, 原书P102-P103给出了以图5.7中隐层到输出层的连接权$w_{hj}$为例的推导过程, 输入层到隐层的连接权$v_{ih}$的推导过程可以照葫芦画瓢完成推导.

## 5.4 试述式(5.6)中学习率的取值对神经网络训练的影响

答: 学习率$\eta$类似于梯度下降过程的步长, 太大会导致误差函数在局部最低点来回震荡, 无法收敛. 太小则学习效率太低收敛过慢, 而且容易陷入局部最低点, 跳不出去.

## 5.5 试编程实现标准BP算法和累积BP算法, 在西瓜数据集3.0 上分别用这两个算法训练一个单隐层网络, 并进行比较.

### 标准BP算法

标准BP算法在西瓜数据集3.0上以学习率$\eta=0.1$训练20轮：

```bash
$ python3 -m neural_networks.back_propagation
输入-训练集:
       密度    含糖率
编号              
1   0.697  0.460
2   0.774  0.376
3   0.634  0.264
4   0.608  0.318
5   0.556  0.215
6   0.403  0.237
7   0.481  0.149
8   0.437  0.211
9   0.666  0.091
10  0.243  0.267
11  0.245  0.057
12  0.343  0.099
13  0.639  0.161
14  0.657  0.198
15  0.360  0.370
16  0.593  0.042
17  0.719  0.103

输入-标记集:
   好瓜
编号   
1   1
2   1
3   1
4   1
5   1
6   1
7   1
8   1
9   0
10  0
11  0
12  0
13  0
14  0
15  0
16  0
17  0
已训练0轮, 累积误差: 0.1531
当前网络:
第2层1个神经元(阈值, 连接权值):
[0.011]
[[0.988]
 [0.879]]
第1层2个神经元(阈值, 连接权值):
[0.646 0.15 ]
[[0.875 0.334]
 [0.759 0.611]]
第0层2个神经元(阈值, 连接权值):
[0 0]

已训练1轮, 累积误差: 0.1459
当前网络:
第2层1个神经元(阈值, 连接权值):
[0.1]
[[0.947]
 [0.833]]
第1层2个神经元(阈值, 连接权值):
[0.667 0.169]
[[0.865 0.325]
 [0.757 0.609]]
第0层2个神经元(阈值, 连接权值):
[0 0]

...


已训练20轮, 累积误差: 0.1220
当前网络:
第2层1个神经元(阈值, 连接权值):
[0.735]
[[0.723]
 [0.552]]
第1层2个神经元(阈值, 连接权值):
[0.796 0.281]
[[0.825 0.289]
 [0.776 0.624]]
第0层2个神经元(阈值, 连接权值):
[0 0]
```

### 累积BP算法

累积BP算法在西瓜数据集3.0上训练20轮：

```bash
$ python3 -m neural_networks.accumulated_back_propagation
输入-训练集:
       密度    含糖率
编号              
1   0.697  0.460
2   0.774  0.376
3   0.634  0.264
4   0.608  0.318
5   0.556  0.215
6   0.403  0.237
7   0.481  0.149
8   0.437  0.211
9   0.666  0.091
10  0.243  0.267
11  0.245  0.057
12  0.343  0.099
13  0.639  0.161
14  0.657  0.198
15  0.360  0.370
16  0.593  0.042
17  0.719  0.103

输入-标记集:
   好瓜
编号   
1   1
2   1
3   1
4   1
5   1
6   1
7   1
8   1
9   0
10  0
11  0
12  0
13  0
14  0
15  0
16  0
17  0
已训练0轮, 累积误差: 0.1241
当前网络:
第2层1个神经元(阈值, 连接权值):
[0.395]
[[0.314]
 [0.566]]
第1层2个神经元(阈值, 连接权值):
[0.848 0.01 ]
[[0.025 0.25 ]
 [0.615 0.541]]
第0层2个神经元(阈值, 连接权值):
[0 0]

已训练1轮, 累积误差: 0.1241
当前网络:
第2层1个神经元(阈值, 连接权值):
[0.396]
[[0.314]
 [0.566]]
第1层2个神经元(阈值, 连接权值):
[0.848 0.01 ]
[[0.025 0.25 ]
 [0.615 0.541]]
第0层2个神经元(阈值, 连接权值):
[0 0]

...

已训练20轮, 累积误差: 0.1240
当前网络:
第2层1个神经元(阈值, 连接权值):
[0.412]
[[0.311]
 [0.56 ]]
第1层2个神经元(阈值, 连接权值):
[0.849 0.012]
[[0.025 0.25 ]
 [0.616 0.542]]
第0层2个神经元(阈值, 连接权值):
[0 0]
```

### 标准BP算法和累积BP算法比较

标准BP算法每次针对单个训练样例更新权值与阈值，参数更新频繁, 不同样例可能抵消, 需要多次迭代；累积BP算法其优化目标是最小化整个训练集上的累计误差读取整个训练集一遍才对参数进行更新, 参数更新频率较低。在很多任务中, 累计误差下降到一定程度后, 进一步下降会非常缓慢, 这时标准BP算法往往会获得较好的解, 尤其当训练集非常大时效果更明显.

以本题的例子，在西瓜数据集3.0，大约需要训练1000-2000轮可以看出二种算法收敛速度有明显差距.

### 修改训练参数

编辑[accumulated_back_propagation.py](../neural_networks/accumulated_back_propagation.py)和[back_propagation.py](../neural_networks/back_propagation.py)的main方法可以修改训练参数:

```python
    config = {CONST_CONFIG_KEY_TIMES: 20} # 停止条件是训练20轮
    leaning_rate = 0.1 # 学习率
```
