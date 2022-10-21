# xiguashu
这里存放我学习周志华老师“西瓜书“《机器学习》课后习题中的编程题.

## 设置运行环境
TODO

## 第四章 决策树

4.3 试编程实现基于信息嫡进行划分选择的决策树算法，并为表4.3中数据生成一棵决策树.  
这里实现了ID3和C4.5二种决策树算法, 代码中使用的是C4.5算法.  
decision_tree.py: 二种决策树算法的实现.  
decision_tree_test.py: decision_tree对应的单元测试用例.  
西瓜数据集 2.0.csv: 对应书中P76 表4.1 西瓜数据集 2.0, CSV格式.  
西瓜数据集 3.0.csv: 对应书中P84 表4.4 西瓜数据集 3.0, CSV格式.  

```bash
python /Users/liyue/Workspace/xiguashu/chapter_4/decision_tree.py
输入-数据集 D:
TrainingSet: label_name=好瓜, samples(17):
    色泽  根蒂  敲声  纹理  脐部  触感     密度    含糖率 好瓜
编号                                         
1   青绿  蜷缩  浊响  清晰  凹陷  硬滑  0.697  0.460  是
2   乌黑  蜷缩  沉闷  清晰  凹陷  硬滑  0.774  0.376  是
3   乌黑  蜷缩  浊响  清晰  凹陷  硬滑  0.634  0.264  是
4   青绿  蜷缩  沉闷  清晰  凹陷  硬滑  0.608  0.318  是
5   浅白  蜷缩  浊响  清晰  凹陷  硬滑  0.556  0.215  是
6   青绿  稍蜷  浊响  清晰  稍凹  软粘  0.403  0.237  是
7   乌黑  稍蜷  浊响  稍糊  稍凹  软粘  0.481  0.149  是
8   乌黑  稍蜷  浊响  清晰  稍凹  硬滑  0.437  0.211  是
9   乌黑  稍蜷  沉闷  稍糊  稍凹  硬滑  0.666  0.091  否
10  青绿  硬挺  清脆  清晰  平坦  软粘  0.243  0.267  否
11  浅白  硬挺  清脆  模糊  平坦  硬滑  0.245  0.057  否
12  浅白  蜷缩  浊响  模糊  平坦  软粘  0.343  0.099  否
13  青绿  稍蜷  浊响  稍糊  凹陷  硬滑  0.639  0.161  否
14  浅白  稍蜷  沉闷  稍糊  凹陷  硬滑  0.657  0.198  否
15  乌黑  稍蜷  浊响  清晰  稍凹  软粘  0.360  0.370  否
16  浅白  蜷缩  浊响  模糊  平坦  硬滑  0.593  0.042  否
17  青绿  蜷缩  沉闷  稍糊  稍凹  硬滑  0.719  0.103  否

输入-属性集 A:
{Continuous attribute: name=含糖率, Discrete attribute: name=根蒂, values={'蜷缩', '硬挺', '稍蜷'}, Discrete attribute: name=脐部, values={'凹陷', '稍凹', '平坦'}, Discrete attribute: name=色泽, values={'浅白', '青绿', '乌黑'}, Discrete attribute: name=触感, values={'软粘', '硬滑'}, Continuous attribute: name=密度, Discrete attribute: name=敲声, values={'清脆', '浊响', '沉闷'}, Discrete attribute: name=纹理, values={'模糊', '稍糊', '清晰'}}

输出-决策树:
Classify(分类属性): 纹理, children(3):
|--纹理=模糊: Label(标记值): 否
|--纹理=稍糊: Classify(分类属性): 触感, children(2):
  |--触感=软粘: Label(标记值): 是
  |--触感=硬滑: Label(标记值): 否
|--纹理=清晰: Classify(分类属性): 密度, children(2):
  |--密度=0.382-: Label(标记值): 否
  |--密度=0.382+: Label(标记值): 是
```
