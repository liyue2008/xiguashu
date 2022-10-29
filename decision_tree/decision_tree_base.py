#-*-coding:utf-8-*- 
from collections import Counter
import copy
from typing import Any
from typing_extensions import Self
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

        

class DataSet:
    """DataSet表示一个训练集，包含训练数据集和数据的标记。

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
        return "DataSet: label_name=%s, samples(%d):\n%s" % (self.label_name, len(self.samples), self.samples)
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
            ret[attr_value] = DataSet(self.samples[self.samples[a.name] == attr_value], self.label_name)
        return ret
    def bi_partition_by_attr(self, attr_name: str, t: float):
        """二分法计算训练集 D 中在属性 attr_name 上,基于最佳划分点t划分的二个子集.

        Returns
        -------
        返回包含二个元素的Dictionary: {'t-': D在属性a上取值不大于t的子集 D-, 't+': D在属性a上取值大于t的子集 D+).
        """
        return {
            ('%.3f-' % t): DataSet(self.samples[self.samples[attr_name] <= t], self.label_name),
            ('%.3f+' % t): DataSet(self.samples[self.samples[attr_name] > t], self.label_name)
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
    def __copy__(self):
        return DecisionTreeNode(self.is_leaf, self.classify_name, self.children, self.label, self.level)

    def __deepcopy__(self, memo):
        return DecisionTreeNode(self.is_leaf, copy.deepcopy(self.classify_name, memo), copy.deepcopy(self.children, memo), copy.deepcopy(self.label, memo), self.level)

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
    def inference(self, test_set: DataSet) -> DataSet:
        """用决策树在数据集test_set上推理, 返回带标记的数据集.
        Parameters
        ----------
        test_set: DataSet
            用于推理的测试数据集.
        
        Returns
        ----------
        返回带标记的数据集.
        """
        ret_set = DataSet(test_set.samples.copy(), test_set.label_name)
        for index, row in ret_set.samples.iterrows():
            ret_set.samples.at[index, ret_set.label_name] = self.inference(row)
        return ret_set

    def inference(self, sample: pd.Series) -> str:
        """用决策树在数据sample上推理, 返回带标记的数据集.
        Parameters
        ----------
        sample: Series
            用于推理的数据.
        Returns
        ----------
            推理出来的分类标记.
        """
        current_node = self
        
        while not current_node.is_leaf:
            classify_value = sample[current_node.classify_name]
            # 如果找不到会抛出KeyError
            current_node = current_node.children[classify_value]
        return current_node.label

    def error_rate(self, validation_set: DataSet) -> float:
        """用决策树在标记的数据集validation_set上推理并计算错误率.
        Parameters
        ----------
        validation_set: DataSet
            用于推理的数据集.
        
        Returns
        ----------
        返回带标记的数据集.
        """
        error_count = 0
        for index, row in validation_set.samples.iterrows():
            prediction_value = self.inference(row)
            if prediction_value != row[validation_set.label_name]:
                error_count = error_count + 1
        error_rate = error_count / validation_set.len()
        return error_rate
    def accuracy(self, validation_set: DataSet) -> float:
        """用决策树在标记的数据集validation_set上推理并计算精度.
        """
        return 1 - self.error_rate(validation_set)


def tree_generate(D: DataSet, A: set, select_partition_method, node_level_in_tree: int = 0) -> DecisionTreeNode : 
    """决策树学习基本算法的实现

    Parameters
    ----------
    D : DataSet
        训练集 D = {(x1, y1), (x2, y2), ... , (xm, ym)};
    A : set of Attribute
        属性集 A = {a1, a2, ... , ad};
    select_partition_method: method
        这个参数是一个函数, 功能是: 从A中选择最优划分属性a*, 并以最优划分属性对 D 进行分区.
        函数定义:
        def select_partition_method(D: DataSet, A: set) -> Tuple[Attribute, dict]:
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
    node_level_in_tree: int
        节点在树中的层级, 根节点取值为0. 
        只用于绘制或打印树时布局节点.
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
    training_set_labels = D.samples[D.label_name].to_numpy()
    if (training_set_labels[0] == training_set_labels).all():
        C = training_set_labels[0]
        node.label = C
        node.is_leaf = True
        return node
 
    # (2)当前属性集为空，或是所有样本在所有属性上取值相同，无法划分;
    # 5: if A = 空 OR D中样本在A上取值相同 then
    # 6:    将 node 标记为叶子节点，其类别标记为D中样本最多的类; return
    # 7: end if
    node.label = majority_in_list(D.samples[D.label_name].to_list())
    if(len(A) == 0 or D.all_samples_same(A)):
        node.is_leaf = True
        return node
    
    #8: 从A中选择最优划分属性a*(classify_attribute), 并以最优划分属性对 D 进行分区.
    
    # Dv表示 D 中在 a* 上取值为 av* 的样本子集, Dv_dict 是所有Dv的集合.
    # key为最优划分属性的取值av*(classify_value), value为 Dv
    (classify_attribute, Dv_dict) = select_partition_method(D, A)
    node.classify_name = classify_attribute.name

    # 9: for a* 的每一个值 av* do
    # 10:   为 node 生成一个分支；另Dv表示 D 中在 a* 上取值为 av* 的样本子集；
    # 11:   if Dv 为空 then   
    # 12:       将分支节点标记为叶节点，其类别标记为D中样本最多的类; return
    # 13:   else
    # 14:       以 tree_generate(Dv, A\{a*}) 为分支节点
    # 15:   end if
    # 16:end for

    
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
            childNode = tree_generate(Dv, sub_a, select_partition_method, node_level_in_tree + 1)
        node.children[classify_value] = childNode
            
    return node

