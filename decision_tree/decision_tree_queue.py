#-*-coding:utf-8-*- 
from .decision_tree_base import *
from .gain import select_partition_method_gain
import queue

class TreeGenerateQueueItem:
    def __init__(self, node: DecisionTreeNode, training_set: DataSet, attributes: set) -> None:
        """用于生成决策树过程中,保持生成中的子节点 node, 及其对应的训练集 training_set 和属性集 attributes
        """
        self.node = node
        self.training_set = training_set
        self.attributes = attributes

def tree_generate_queue(training_set: DataSet, attributes: set, select_partition_method, max_depth: int = 0) -> DecisionTreeNode : 
    """使用队列实现的决策树学习基本算法

    Parameters
    ----------
    training_set : DataSet
        训练集 D = {(x1, y1), (x2, y2), ... , (xm, ym)};
    attributes : set of Attribute
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
    max_depth: int
        参数max_depth控制树的最大深度 
    Returns
    -------
    node: DecisionTreeNode
        以node为根节点的一棵决策树。
    """
    
    node_queue = queue.Queue()
    tree = DecisionTreeNode(children= {})
    node_queue.put(TreeGenerateQueueItem(tree, training_set, attributes))
    # 广度遍历优先, 逐层生成树节点.
    # 先将根节点入队.
    # 依次从队中取出节点处理, 知道队列为空则处理完成
    #   判断当前节点是否有子节点, 如果有, 则将这些子节点入队
    while not node_queue.empty():
        # 1: 生成节点node
        item = node_queue.get()
        node = item.node
        D = item.training_set
        A = item.attributes
        # (1)当前结点包含的样本全属于同一类别，无需划分;
        # 2: if D 中样本全属于同一类别C then
        # 3:   将 node 标记为 C 类叶节点；return；
        # 4: end if
        training_set_labels = D.samples[D.label_name].to_numpy()
        if (training_set_labels[0] == training_set_labels).all():
            C = training_set_labels[0]
            node.label = C
            node.is_leaf = True
            continue
    
        # (2)当前属性集为空，或是所有样本在所有属性上取值相同，无法划分;
        # 5: if A = 空 OR D中样本在A上取值相同 then
        # 6:    将 node 标记为叶子节点，其类别标记为D中样本最多的类; return
        # 7: end if
        node.label = majority_in_list(D.samples[D.label_name].to_list())
        if len(A) == 0 or D.all_samples_same(A) or (max_depth > 0 and node.depth >= max_depth):
            node.is_leaf = True
            continue
        
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
            childNode = DecisionTreeNode(depth = node.depth + 1, children = {})
            if Dv.samples.empty:
                childNode.is_leaf = True
                childNode.label = majority_in_list(D.samples[D.label_name].to_list())
            else:
                sub_a = A.copy()
                sub_a.remove(classify_attribute)
                node_queue.put(TreeGenerateQueueItem(childNode, Dv, sub_a))
            node.children[classify_value] = childNode 
    return tree

def tree_generate_gain(D: DataSet, A: set) -> DecisionTreeNode:
    return tree_generate_queue(D, A, select_partition_method_gain)

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