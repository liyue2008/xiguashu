#-*-coding:utf-8-*- 
from .decision_tree_base import *

def tree_generate_prepruning(training_set: DataSet, test_set: DataSet, A: set, select_partition_method, node_level_in_tree: int = 0) -> DecisionTreeNode : 
    """带预剪枝的决策树算法实现

    Parameters
    ----------
    training_set : DataSet
        训练集 D = {(x1, y1), (x2, y2), ... , (xm, ym)};
    test_set : DataSet
        测试集;
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
    # 1: 生成节点node
    node = DecisionTreeNode(level= node_level_in_tree, children= {})
    # (1)当前结点包含的样本全属于同一类别，无需划分;
    # 2: if D 中样本全属于同一类别C then
    # 3:   将 node 标记为 C 类叶节点；return；
    # 4: end if
    training_set_labels = training_set.samples[training_set.label_name].to_numpy()
    if (training_set_labels[0] == training_set_labels).all():
        C = training_set_labels[0]
        node.label = C
        node.is_leaf = True
        return node
 
    # (2)当前属性集为空，或是所有样本在所有属性上取值相同，无法划分;
    # 5: if A = 空 OR D中样本在A上取值相同 then
    # 6:    将 node 标记为叶子节点，其类别标记为D中样本最多的类; return
    # 7: end if
    if(len(A) == 0 or training_set.all_samples_same(A)):
        node.is_leaf = True
        node.label = majority_in_list(training_set.samples[training_set.label_name].to_list())
        return node
    
    #8: 从A中选择最优划分属性a*(classify_attribute), 并以最优划分属性对 D 进行分区.
    
    # Dv表示 D 中在 a* 上取值为 av* 的样本子集, Dv_dict 是所有Dv的集合.
    # key为最优划分属性的取值av*(classify_value), value为 Dv
    (classify_attribute, Dv_dict) = select_partition_method(training_set, A)
    node.classify_name = classify_attribute.name

    # 9: for a* 的每一个值 av* do
    # 10:   为 node 生成一个分支；另Dv表示 D 中在 a* 上取值为 av* 的样本子集；
    # 11:   if Dv 为空 then   
    # 12:       将分支节点标记为叶节点，其类别标记为D中样本最多的类; return
    # 13:   else
    # 14:       以 tree_generate(Dv, A\{a*}) 为分支节点
    # 15:   end if
    # 16:end for

    # 构造划分前的决策树
    before = copy.deepcopy(node)
    before.is_leaf = True
    before.label = majority_in_list(majority_in_list(training_set.samples[training_set.label_name].to_list()))

    # 构造划分后的决策树
    after = copy.deepcopy(node)
    for classify_value in Dv_dict:
        Dv = Dv_dict[classify_value]
        childNode = DecisionTreeNode(level = node_level_in_tree + 1)
        childNode.is_leaf = True
        childNode.label = majority_in_list(training_set.samples[training_set.label_name].to_list())
        after.children[classify_value] = childNode

    # 在测试集上计算划分前和划分后的精度
    accuracy_before = before.accuracy(test_set)
    accuracy_after = after.accuracy(test_set)
    # 决策是否预剪枝
    is_prepruning =  accuracy_before > accuracy_after

    if is_prepruning:
        node.is_leaf = True
        node.label = majority_in_list(training_set.samples[training_set.label_name].to_list())
    else:
        for classify_value in Dv_dict:
            Dv = Dv_dict[classify_value]
            if Dv.samples.empty:
                childNode = DecisionTreeNode(level = node_level_in_tree + 1)
                childNode.is_leaf = True
                childNode.label = majority_in_list(training_set.samples[training_set.label_name].to_list())
            else:
                sub_a = A.copy()
                sub_a.remove(classify_attribute)
                # print('%s=%s:' % (classify_attribute, classify_value))
                childNode = tree_generate_prepruning(Dv, test_set, sub_a, select_partition_method, node_level_in_tree + 1)
            node.children[classify_value] = childNode
            
    return node