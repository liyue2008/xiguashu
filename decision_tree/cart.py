#-*-coding:utf-8-*- 
from .decision_tree_base import *

def gini(D: TrainingSet) -> float:
    """计算样本集合 D 的基尼值
    
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
    sum_pk_square = 0
    for count in df_count_by_class:
        pk = count / D.len()
        pk_square = pk * pk
        sum_pk_square += pk_square
    return 1 - sum_pk_square

def gini_index(D: TrainingSet, a: Attribute) -> float:
    """计算样本 D 在离散值属性 a 上的信息增益(information entropy).
    """
    # print("Training set:\n%s\n" % D)
    len_of_D = D.len()
    # print('len_of_D: %.3f\n' % len_of_D)
    gini_index = 0
    for Dv in D.partition_by_attr(a):
        gini_index += ( Dv.len() / len_of_D ) * gini(Dv)
    return gini_index   

def select_partition_method_gini_index(D: TrainingSet, A: set) -> Tuple[Attribute, dict]:
    """用基尼指数来进行决策树的划分属性选择, 选择属性a* = arg min Gini_index(D,a), a ∈ A.

    Parameters
    ----------
    D : TrainingSet
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
    classify_attribute_gini_index = sys.float_info.max
    for a in A:
        gain_index_a = gini_index(D, a);
        if gain_index_a < classify_attribute_gini_index:
            classify_attribute_gini_index = gain_index_a
            classify_attribute = a
    Dv_dict = D.partition_by_attr(classify_attribute)
    return (classify_attribute, Dv_dict)

def tree_generate_CART(D: TrainingSet, A: set) -> DecisionTreeNode:
    return tree_generate(D, A, select_partition_method_gini_index)

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
    D = TrainingSet(df, '好瓜')

    print('输入-数据集 D:')
    print(D)
    print('\n输入-属性集 A:')
    print(A)
    tree = tree_generate_CART(D, A)
    print('\n输出-决策树:')
    print(tree)