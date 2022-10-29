#-*-coding:utf-8-*- 
from .decision_tree_base import *
from .gini import select_partition_method_gini_index

def is_prepruning(node: DecisionTreeNode, training_set: DataSet, test_set: DataSet, node_level_in_tree, Dv_dict: dict) -> bool:
    """基于精度判断是否需要预剪枝.

    Parameters
    ----------
    node: DecisionTreeNode
        当前决策树节点.
    training_set : DataSet
        训练集 D = {(x1, y1), (x2, y2), ... , (xm, ym)};
    test_set : DataSet
        测试集;
    node_level_in_tree: int
        节点在树中的层级, 根节点取值为0. 
        只用于绘制或打印树时布局节点.
    Dv_dict: dict
        Dv表示 D 中在 a* 上取值为 av* 的样本子集, Dv_dict 是所有Dv的集合.
        key: str 
            最优划分属性的取值av*.
        value: DataSet
            Dv表示 D 中在 a* 上取值为 av* 的样本子集.
    Returns
    -------
    是否预剪枝, true: 禁止划分(剪枝), false: 划分(不剪枝)。
    """

    # 构造划分前的决策树
    before = copy.deepcopy(node)
    before.is_leaf = True
    before.label = majority_in_list(training_set.samples[training_set.label_name].to_list())


    # 构造划分后的决策树
    after = copy.deepcopy(node)
    for classify_value in Dv_dict:

        Dv = Dv_dict[classify_value]
        childNode = DecisionTreeNode(level = node_level_in_tree + 1)

        training_set_labels = training_set.samples[training_set.label_name].to_numpy()
        if Dv.len() == 0:
            childNode.label = majority_in_list(training_set.samples[Dv.label_name].to_list())
        elif training_set_labels[0] == training_set_labels.all():
            childNode.label = training_set_labels[0]
        else:
            childNode.label = majority_in_list(Dv.samples[Dv.label_name].to_list())
        childNode.is_leaf = True
        
        after.children[classify_value] = childNode

    # 在测试集上计算划分前和划分后的精度
    accuracy_before = before.accuracy(test_set)
    accuracy_after = after.accuracy(test_set)
    # print('分类属性: %s' % node.classify_name)
    # print('before: acc = %.3f, tree:\n%s' % (accuracy_before, str(before)))
    # print('after: acc = %.3f, tree:\n%s\n' % (accuracy_after, str(after)))
    # 决策是否预剪枝
    return  accuracy_before >= accuracy_after

def is_postpruning(node: DecisionTreeNode, test_set: DataSet) -> bool:
    """基于精度判断是否需要后剪枝.

    Parameters
    ----------
    node: DecisionTreeNode
        当前决策树节点.
    test_set : DataSet
        测试集;
    Returns
    -------
    是否剪枝, true: 禁止划分(剪枝), false: 划分(不剪枝)。
    """

    # 构造剪枝后的决策树
    postpruning_node = copy.deepcopy(node)
    postpruning_node.is_leaf = True
    postpruning_node.children.clear()
    postpruning_node.classify_name = ''


    # 在测试集上计算划分前和划分后的精度
    accuracy_original = node.accuracy(test_set)
    accuracy_postpruning = postpruning_node.accuracy(test_set)
    
    # print('original: acc = %.3f, tree:\n%s' % (accuracy_original, str(node)))
    # print('postpruning: acc = %.3f, tree:\n%s\n' % (accuracy_postpruning, str(postpruning_node)))
    # 决策是否预剪枝
    return  accuracy_postpruning >= accuracy_original


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
            value: DataSet
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
    
    # 决策是否预剪枝
    prepruning =  is_prepruning(node, training_set, test_set, node_level_in_tree, Dv_dict)

    # 9: for a* 的每一个值 av* do
    # 10:   为 node 生成一个分支；另Dv表示 D 中在 a* 上取值为 av* 的样本子集；
    # 11:   if Dv 为空 then   
    # 12:       将分支节点标记为叶节点，其类别标记为D中样本最多的类; return
    # 13:   else
    # 14:       以 tree_generate(Dv, A\{a*}) 为分支节点
    # 15:   end if
    # 16:end for

    if prepruning:
        node.is_leaf = True
        node.label = majority_in_list(training_set.samples[training_set.label_name].to_list())
    else:
        test_set_dict = test_set.partition_by_attr(classify_attribute)
        for classify_value in Dv_dict:
            Dv = Dv_dict[classify_value]
            if Dv.samples.empty:
                childNode = DecisionTreeNode(level = node_level_in_tree + 1)
                childNode.is_leaf = True
                childNode.label = majority_in_list(training_set.samples[training_set.label_name].to_list())
            else:
                sub_a = A.copy()
                sub_a.remove(classify_attribute)
                # print('%s=%s:' % (classify_attribute.name, classify_value))
                # print('训练集: ', Dv)
                # print('测试集: ', test_set_dict[classify_value]) 
                childNode = tree_generate_prepruning(Dv, test_set_dict[classify_value], sub_a, select_partition_method, node_level_in_tree + 1)
            node.children[classify_value] = childNode
            
    return node

def postpruning(node: DecisionTreeNode, test_set: DataSet) -> DecisionTreeNode:
    """用递归对树进行后剪枝

    Parameters
    ----------
    node: DecisionTreeNode
        当前决策树节点.
    test_set : DataSet
        测试集.

    Returns
    -------
    剪枝后的决策树。
    """
    if node.is_leaf:
        return node
    for attr_value in node.children:
        # 获取子树
        child_node = node.children[attr_value]
        # 如果子树非叶子节点, 对子树进行后剪枝
        if not child_node.is_leaf:
            child_test_set = DataSet(test_set.samples[test_set.samples[node.classify_name] == attr_value], test_set.label_name)
            # 先对子节点进行后剪枝, 即自底向上剪枝
            child_node = postpruning(child_node, child_test_set)
    # 判断是否需要后剪枝
    if is_postpruning(node, test_set):
        return DecisionTreeNode(is_leaf = True, label = node.label, level = node.level)
    else:
        return node



def tree_generate_gini_prepruning(training_set: DataSet, test_set: DataSet, A: set) -> DecisionTreeNode:
    return tree_generate_prepruning(training_set, test_set, A, select_partition_method_gini_index)

def tree_generate_gini_postpruning(training_set: DataSet, test_set: DataSet) -> DecisionTreeNode:   
    tree = tree_generate(training_set, test_set, A, select_partition_method_gini_index)
    tree_postpruning = postpruning(tree, test_set)
    return tree_postpruning

def prepruning_main():
    A = {
            Attribute('色泽', {'青绿', '乌黑', '浅白'}),
            Attribute('根蒂', {'稍蜷', '蜷缩', '硬挺'}),
            Attribute('敲声', {'沉闷', '浊响', '清脆'}),
            Attribute('纹理', {'清晰', '稍糊', '模糊'}),
            Attribute('脐部', {'凹陷', '稍凹', '平坦'}),
            Attribute('触感', {'硬滑', '软粘'})
        }
    training_df = pd.read_csv('data/西瓜数据集 2.0 训练集.csv').set_index('编号')
    training_data_set = DataSet(training_df, '好瓜')
    test_df = pd.read_csv('data/西瓜数据集 2.0 验证集.csv').set_index('编号')
    test_data_set = DataSet(test_df, '好瓜')

    print('输入-训练集:')
    print(training_data_set)
    print('输入-验证集:')
    print(test_data_set)
    print('\n输入-属性集 A:')
    print(A)
    tree = tree_generate_gini_prepruning(training_data_set, test_data_set, A)
    print('\n输出-决策树:')
    
    print(tree)
    print('==========')
    test_df = pd.read_csv('data/西瓜数据集 2.0 验证集.csv').set_index('编号')
    test_data_set = DataSet(test_df, '好瓜')
    accuracy = tree.accuracy(test_data_set)
    print('决策树在如下验证集上的精度: %.3f' % accuracy)
    print(test_data_set)

def postpruning_main():
    A = {
            Attribute('色泽', {'青绿', '乌黑', '浅白'}),
            Attribute('根蒂', {'稍蜷', '蜷缩', '硬挺'}),
            Attribute('敲声', {'沉闷', '浊响', '清脆'}),
            Attribute('纹理', {'清晰', '稍糊', '模糊'}),
            Attribute('脐部', {'凹陷', '稍凹', '平坦'}),
            Attribute('触感', {'硬滑', '软粘'})
        }
    training_df = pd.read_csv('data/西瓜数据集 2.0 训练集.csv').set_index('编号')
    training_data_set = DataSet(training_df, '好瓜')
    test_df = pd.read_csv('data/西瓜数据集 2.0 验证集.csv').set_index('编号')
    test_data_set = DataSet(test_df, '好瓜')

    print('输入-训练集:')
    print(training_data_set)
    print('输入-验证集:')
    print(test_data_set)
    print('\n输入-属性集 A:')
    print(A)
    tree = tree_generate_gini_postpruning(training_data_set, test_data_set, A)
    print('\n输出-决策树:')
    
    print(tree)
    print('==========')
    test_df = pd.read_csv('data/西瓜数据集 2.0 验证集.csv').set_index('编号')
    test_data_set = DataSet(test_df, '好瓜')
    accuracy = tree.accuracy(test_data_set)
    print('决策树在如下验证集上的精度: %.3f' % accuracy)
    print(test_data_set)


if __name__ == '__main__':
    prepruning_main()
