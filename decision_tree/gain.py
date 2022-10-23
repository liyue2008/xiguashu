#-*-coding:utf-8-*- 
import math
from typing import Tuple
from .decision_tree_base import *

def ent(D: DataSet) -> float:
    """计算样本集合 D 的信息熵(information entropy)
    
    Parameters
    ----------
    D : DataSet
        样本集合 D。

    Returns
    -------
    样本集合 D 的信息熵(information entropy)
    """
    
    """样本中各分类的数量，
        例如：
        好瓜    数量
        是     2
        否     1
        Name: 好瓜, dtype: int64
    """ 
    df_count_by_class = D.samples.groupby([D.label_name])[D.label_name].count()
    negative_ent = 0
    for count in df_count_by_class:
        pk = count / D.len()
        log2pk = 0 if pk == 0 else math.log2(pk)
        negative_ent += pk * log2pk
    return - negative_ent

def gain_continuous(D: DataSet, a: Attribute) -> Tuple[float, float]:
    """计算样本 D 在连续值属性 a 上, 采用二分法(bi-partition)的最优划分点t, 并返回基于t的二分信息增益.
    """
    # 将属性 a 在训练集 D 上的所有样本取值转换成list并排序
    sorted_values = D.samples[a.name].values.tolist()
    sorted_values.sort()

    # 计算 sorted_values 上每相邻二个点的中位值(候选划分点)集合 Ta

    gain = -1
    t = -1 # 最优划分点
    for i in range(len(sorted_values) - 1):
        ti = (sorted_values[i] + sorted_values[i + 1])/2
        # 复制一份 D, 将 属性 a 的取值按照 t 做二分, 变成{不大于t(Dt-), 大于t(Dt+)}的二分离散值
        Dta = DataSet(D.samples.copy(), D.label_name)
        for j, row in Dta.samples.iterrows():
            Dta.samples.at[j, a.name] = ('%.3f' % ti) + ('-' if row[a.name] <= ti else '+')
        at = Attribute(a.name, {('%.3f-' % ti), ('%.3f+' % ti)})
        gain_D_a_ti = gain_discrete(Dta, at)
        if gain_D_a_ti > gain:
            gain = gain_D_a_ti
            t = ti
    return (gain, t)   


def gain_discrete(D: DataSet, a: Attribute) -> float:
    """计算样本 D 在离散值属性 a 上的信息增益(information entropy).
    """
    # print("Training set:\n%s\n" % D)
    ent_of_D = ent(D)
    # print('ent_of_D: %.3f\n' % ent_of_D)
    len_of_D = D.len()
    # print('len_of_D: %.3f\n' % len_of_D)
    t = 0
    dict = D.partition_by_attr(a)
    for Dv in dict.values():
        t += ( Dv.len() / len_of_D ) * ent(Dv)
        # print('\t%s = %s: %.3f\n%s' % (a.name, av, ( Dv.len() / len_of_D ) * ent(Dv), Dv))
    # print('t: %.3f\n' % t)
    gain = ent_of_D - t
    # print('Gain(%s) = %.3f' % (a.name, gain))
    return gain   

def gain(D: DataSet, a: Attribute) -> float:
    """计算样本 D 在属性 a 上的信息增益(information entropy).
    """
    if a.is_continuous:
        return gain_continuous(D, a)
    else:
        return gain_discrete(D, a)

def select_partition_method_gain(D: DataSet, A: set) -> Tuple[Attribute, dict]:
    """基于ID3决策树算法, 用信息增益来进行决策树的最优划分属性选择, 选择属性a* = arg max Gain(D,a), a ∈ A.
    采用二分法(bi-partition)对连续属性进行处理, 源自C4.5决策树算法.
    
    Parameters
    ----------
    D : DataSet
        训练集 D = {(x1, y1), (x2, y2), ... , (xm, ym)};
    A : set of Attribute
        属性集 A = {a1, a2, ... , ad};

    Returns
    -------
    classify_attribute : Attribute
        信息增益最大的属性.
    Dv_dict: dict
        Dv表示 D 中在 a* 上取值为 av* 的样本子集, Dv_dict 是所有Dv的集合.
        key: str 
            最优划分属性的取值av*.
        value: TrainSet
            Dv表示 D 中在 a* 上取值为 av* 的样本子集.
    """
    classify_attribute = Attribute()
    classify_attribute_gain = 0
    classify_attribute_t = 0
    for a in A:
        gain_a = 0
        t = 0
        if a.is_continuous:
            (gain_a, t) = gain_continuous(D, a)
        else:
            gain_a = gain_discrete(D, a)
        if gain_a > classify_attribute_gain:
            classify_attribute_gain = gain_a
            classify_attribute = a
            classify_attribute_t = t    
    # Dv表示 D 中在 a* 上取值为 av* 的样本子集, Dv_dict 是所有Dv的集合.
    # key为最优划分属性的取值av*(classify_value), value为 Dv
    if classify_attribute.is_continuous:
        Dv_dict = D.bi_partition_by_attr(classify_attribute.name, classify_attribute_t)
    else:
        Dv_dict = D.partition_by_attr(classify_attribute)
    return (classify_attribute, Dv_dict)

def tree_generate_gain(D: DataSet, A: set) -> DecisionTreeNode:
    return tree_generate(D, A, select_partition_method_gain)

if __name__ == '__main__':
    A = {
            Attribute('色泽', {'青绿', '乌黑', '浅白'}),
            Attribute('根蒂', {'稍蜷', '蜷缩', '硬挺'}),
            Attribute('敲声', {'沉闷', '浊响', '清脆'}),
            Attribute('纹理', {'清晰', '稍糊', '模糊'}),
            Attribute('脐部', {'凹陷', '稍凹', '平坦'}),
            Attribute('触感', {'硬滑', '软粘'}),
            Attribute('密度', is_continuous = True),
            Attribute('含糖率', is_continuous = True)
        }
    df = pd.read_csv('data/西瓜数据集 3.0.csv')
    df.set_index('编号', inplace=True)
    D = DataSet(df, '好瓜')

    print('输入-数据集 D:')
    print(D)
    print('\n输入-属性集 A:')
    print(A)
    tree = tree_generate_gain(D, A)
    print('\n输出-决策树:')
    print(tree)