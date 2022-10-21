#-*-coding:utf-8-*- 
import math
from collections import Counter
from typing import Tuple
import pandas as pd

def all_same(items):
    return all(x == items[0] for x in items)

def majority_in_list(l: list):
    """获取list中的大多数(值相同且个数最多的元素的值)。
    """
    c = Counter(l)
    return c.most_common()[0][0]
class Attribute:
    """AttributeSet表示一个样本的一个属性, 包含属性名称和属性的全部类别(取值范围)

    Attributes
    ----------
    name : str
        属性名称;
    values : set
        属性的全部类别(取值范围).
    is_continuous : bool
        属性是离散值(False)还是连续值(True), 默认为False离散值.
    """
    def __init__(self, name: str = '', values: set = set(), is_continuous: bool = False) -> None:
        self.name = name
        self.values = values
        self.is_continuous = is_continuous
    def __str__(self) -> str:
        if self.is_continuous:
            return 'Continuous attribute: name=%s' % self.name
        else:
            return 'Discrete attribute: name=%s, values=%s' % (self.name, self.values)
    def __repr__(self) -> str:
        return self.__str__()

        

class TrainingSet:
    """TrainingSet表示一个训练集，包含训练数据集和数据的标记。

    Attributes
    ----------
    samples : DataFrame
        训练数据集，包含训练样本和标记。
    label_name : str
        在训练数据集samples中，标记列的列名。
    """
    def  __init__(self, samples: pd.DataFrame, label_name: str = '') -> None :
        self.samples = samples
        self.label_name = label_name
    def __eq__(self, other):
        return (self.label_name == other.label_name and self.samples.equals(other.samples))
    def __str__(self) -> str:
        return "TrainingSet: label_name=%s, samples(%d):\n%s" % (self.label_name, len(self.samples), self.samples)
    def __repr__(self) -> str:
        return self.__str__()
    def len(self) -> int:
        return len(self.samples.index)

    def partition_by_attr(self, a: Attribute):
        """将训练集 D 按照离散属性 a 的取值, 为a的每个取值划分为一个子集Da, 并以Dictionary的形式返回这些子集.  
        Returns
        -------
        返回一个Dictionary, key是属性取值, value是对应的Da的子集.

        """
        ret = {}
        for attr_value in a.values:
            ret[attr_value] = TrainingSet(self.samples[self.samples[a.name] == attr_value], self.label_name)
        return ret
    def bi_partition_by_attr(self, attr_name: str, t: float):
        """二分法计算训练集 D 中在属性 attr_name 上,基于最佳划分点t划分的二个子集.

        Returns
        -------
        返回包含二个元素的Dictionary: {'t-': D在属性a上取值不大于t的子集 D-, 't+': D在属性a上取值大于t的子集 D+).
        """
        return {
            ('%.3f-' % t): TrainingSet(self.samples[self.samples[attr_name] <= t], self.label_name),
            ('%.3f+' % t): TrainingSet(self.samples[self.samples[attr_name] > t], self.label_name)
            }
    def all_samples_same(self, A: set) -> bool:
        """训练集D上是否所有样本在属性列表A中所有属性上取值相同

        Parameters
        ----------
        A : set
            属性集 A = {a1, a2, ... , ad}.

        Returns
        -------
        true: D中样本在A上取值相同。
        false: D中样本在A上取值不完全相同。
        """
        for a in A:
            if (not all_same(self.samples[a.name].values)):
                return False
        return True

class DecisionTreeNode:
    """DecisionTreeNode表示一个一个决策树的节点和它的所有子节点，实际上也是一个决策树。

    Attributes
    ----------
    is_leaf : bool
        是否叶子节点.
    classify_name : str
        最优划分属性名称，例如：‘色泽’;
    children : dict
        key : str
            分类(最优划分属性值)，例如：'青绿'；
        value : DecisionTreeNode
            子节点
        仅当非叶子节点时有效, 表示全部子节点.
    label : str
        仅当叶子节点时有效, 表示分类标记.
    """
    def __init__(self, is_leaf: bool = False, classify_name: str = '', children: dict = {}, label: str = '', level: int = 0) -> None:
        self.is_leaf = is_leaf
        self.children = children
        self.classify_name = classify_name
        self.label = label
        self.level = level
    def __str__(self) -> str:
        summary = ''
        # print(indent + summary)
        children_str = ''
        if not self.is_leaf:
            summary += 'Classify(分类属性): %s' % (self.classify_name)

            children_str += ', children(%d):' % len(self.children)
            for c in self.children:
                indent = ''
                for x in range(self.level):
                    indent += '  '
                indent += '|--'
                children_str += '\n' + indent + '%s=%s: ' % (self.classify_name, c) + str(self.children[c])
        else:
            summary = 'Label(标记值): %s' % (self.label)

        return summary + children_str
    def dict(self):
        """转换树转换为dict, 用于打印输出或绘图.
        """
        # print('leaf node: %r, classify_name: %s, children: %d, label: %s' % (self.is_leaf, self.classify_name, len(self.children), self.label))
        if self.is_leaf:
            return self.label
        else:
            formated_children_dict = dict()
            for key in self.children:
                formated_children_dict[key] = self.children[key].dict()
            return {self.classify_name: formated_children_dict}


def ent(D: TrainingSet) -> float:
    """计算样本集合 D 的信息熵(information entropy)
    
    Parameters
    ----------
    D : TrainingSet
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

def gain_continuous(D: TrainingSet, a: Attribute) -> Tuple[float, float]:
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
        Dta = TrainingSet(D.samples.copy(), D.label_name)
        for j, row in Dta.samples.iterrows():
            Dta.samples.at[j, a.name] = ('%.3f' % ti) + ('-' if row[a.name] <= ti else '+')
        at = Attribute(a.name, {('%.3f-' % ti), ('%.3f+' % ti)})
        gain_D_a_ti = gain_discrete(Dta, at)
        if gain_D_a_ti > gain:
            gain = gain_D_a_ti
            t = ti
    return (gain, t)   


def gain_discrete(D: TrainingSet, a: Attribute) -> float:
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

def gain(D: TrainingSet, a: Attribute) -> float:
    """计算样本 D 在属性 a 上的信息增益(information entropy).
    """
    if a.is_continuous:
        return gain_continuous(D, a)
    else:
        return gain_discrete(D, a)

def selet_best_attribute_method_ID3(D: TrainingSet, A: set) -> Tuple[Attribute, float]:
    """ID3决策树的分类选择算法.
    用信息增益来进行决策树的划分属性选择, 选择属性a* = arg max Gain(D,a).

    Parameters
    ----------
    D : TrainingSet
        训练集 D = {(x1, y1), (x2, y2), ... , (xm, ym)};
    A : set of Attribute
        属性集 A = {a1, a2, ... , ad};

    Returns
    -------
    best_attribute : Attribute
        基于ID3选择的信息增益最大的属性.
    """
    max_attr = Attribute()
    max_attr_gain = 0
    for a in A:
        gain_a = gain_discrete(D, a);
        if gain_a > max_attr_gain:
            max_attr_gain = gain_a
            max_attr = a
    # print('Attr: %s, gain: %.3f' % (max_attr.name, max_attr_gain))
    # print(A)
    # print(D)
    return (max_attr, 0)

def selet_best_attribute_method_C4dot5(D: TrainingSet, A: set) -> Tuple[Attribute, float]:
    """采用二分法(bi-partition)对连续属性进行处理，这正是C 4.5决策树算法.
    用信息增益来进行决策树的划分属性选择, 选择属性a* = arg max Gain(D,a).

    Parameters
    ----------
    D : TrainingSet
        训练集 D = {(x1, y1), (x2, y2), ... , (xm, ym)};
    A : set of Attribute
        属性集 A = {a1, a2, ... , ad};

    Returns
    -------
    best_attribute : Attribute
        基于ID3选择的信息增益最大的属性.
    """
    max_attr = Attribute()
    max_attr_gain = 0
    max_attr_t = 0
    for a in A:
        gain_a = 0
        t = 0
        if a.is_continuous:
            (gain_a, t) = gain_continuous(D, a)
        else:
            gain_a = gain_discrete(D, a)
        if gain_a > max_attr_gain:
            max_attr_gain = gain_a
            max_attr = a
            max_attr_t = t
    # print('Attr: %s, gain: %.3f' % (max_attr.name, max_attr_gain))
    # print(A)
    # print(D)
    return (max_attr, max_attr_t)


def tree_generate(D: TrainingSet, A: set, selet_best_attribute_method, node_level_in_tree: int = 0) -> DecisionTreeNode : 
    """决策树学习基本算法的实现

    Parameters
    ----------
    D : TrainingSet
        训练集 D = {(x1, y1), (x2, y2), ... , (xm, ym)};
    A : set of Attribute
        属性集 A = {a1, a2, ... , ad};
    selet_best_attribute_method: method
        从A中选择最优划分属性a*的方法.

    Returns
    -------
    node: DecisionTreeNode
        以node为根节点的一棵决策树。
    """


    # print(node_level_in_tree, D, A)
    # TODO: 检查训练集D是否为空。
    # 1: 生成节点node
    node = DecisionTreeNode(level= node_level_in_tree, children= {})
    # (1)当前结点包含的样本全属于同一类别，无需划分;
    # 2: if D 中样本全属于同一类别C then
    # 3:   将 node 标记为 C 类叶节点；return；
    # 4: end if
    labels = D.samples[D.label_name].to_numpy()
    if (labels[0] == labels).all():
        C = labels[0]
        node.label = C
        node.is_leaf = True
        return node
 
    # (2)当前属性集为空，或是所有样本在所有属性上取值相同，无法划分;
    # 5: if A = 空 OR D中样本在A上取值相同 then
    # 6:    将 node 标记为叶子节点，其类别标记为D中样本最多的类; return
    # 7: end if
    if(len(A) == 0 or D.all_samples_same(A)):
        node.is_leaf = True
        node.label = majority_in_list(D.labels)
        return node
    
    #8: 从A中选择最优划分属性a*(best_attribute)
    (classify_attribute, t) = selet_best_attribute_method(D, A)
    node.classify_name = classify_attribute.name
    # 9: for a* 的每一个值 av* do
    # 10:   为 node 生成一个分支；另Dv表示 D 中在 a* 上取值为 av* 的样本子集；
    # 11:   if Dv 为空 then   
    # 12:       将分支节点标记为叶节点，其类别标记为D中样本最多的类; return
    # 13:   else
    # 14:       以 tree_generate(Dv, A\{a*}) 为分支节点
    # 15:   end if
    # 16:end for

    if classify_attribute.is_continuous:
        Dv_dict = D.bi_partition_by_attr(classify_attribute.name, t)
    else:
        Dv_dict = D.partition_by_attr(classify_attribute)
    
    for classify_value in Dv_dict:
        Dv = Dv_dict[classify_value]
        if Dv.samples.empty:
            childNode = DecisionTreeNode(level = node_level_in_tree + 1)
            childNode.is_leaf = True
            childNode.label = majority_in_list(D.samples[D.label_name].to_list())
        else:
            sub_a = A.copy()
            sub_a.remove(classify_attribute)
            # print('%s=%s:' % (classify_attribute, classify_value))
            childNode = tree_generate(Dv, sub_a, selet_best_attribute_method, node_level_in_tree + 1)
        node.children[classify_value] = childNode
            
    return node

def tree_generate_ID3(D: TrainingSet, A: set) -> DecisionTreeNode:
    return tree_generate(D, A, selet_best_attribute_method_ID3)
def tree_generate_C4dot5(D: TrainingSet, A: set) -> DecisionTreeNode:
    return tree_generate(D, A, selet_best_attribute_method_C4dot5)


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
    df = pd.read_csv('chapter_4/西瓜数据集 3.0.csv')
    df.set_index('编号', inplace=True)
    D = TrainingSet(df, '好瓜')

    print('输入-数据集 D:')
    print(D)
    print('\n输入-属性集 A:')
    print(A)
    tree = tree_generate_C4dot5(D, A)
    print('\n输出-决策树:')
    print(tree)