# 周志华老师"西瓜书"《机器学习》课后习题中的编程题

![西瓜书](data/xiguashu.jpg)
这里存放我学习周志华老师“西瓜书“《机器学习》课后习题中的编程题.

## 设置运行环境

本项目使用[Python3\(3.10.8\)](https://www.python.org/downloads/)开发, 在设置项目前请确保已正确按照和配置了Python3. 建议使用[venv](https://docs.python.org/zh-cn/3/library/venv.html)进行项目环境隔离.

克隆项目到本地:  

```bash
 ~/temp $ git clone https://github.com/liyue2008/xiguashu.git
Cloning into 'xiguashu'...
remote: Enumerating objects: 83, done.
remote: Counting objects: 100% (83/83), done.
remote: Compressing objects: 100% (53/53), done.
remote: Total 83 (delta 39), reused 66 (delta 25), pack-reused 0
Receiving objects: 100% (83/83), 29.17 KiB | 489.00 KiB/s, done.
Resolving deltas: 100% (39/39), done.
```

进入克隆目录, 创建并激活venv:

```bash
 ~/temp $ cd xiguashu
 ~/temp/xiguashu $ python3 -m venv .
 ~/temp/xiguashu $ ls
LICENSE          data             lib              tests
README.md        decision_tree    pyvenv.cfg
bin              include          requirements.txt
 ~/temp/xiguashu $ source bin/activate
(xiguashu)  ~/temp/xiguashu $ # 注意到命令提示符变了, 有个(xiguashu)前缀, 表示当前环境是一个名为xiguashu的Python环境
```

安装项目所需的第三方包:

```bash
(xiguashu)  ~/temp/xiguashu $ pip3 install -r requirements.txt
```

至此环境已经设置好了. 执行一个单元测试验证一下:

```bash
(xiguashu) ~/temp/xiguashu $ python3 tests/decision_tree_test.py
.........
----------------------------------------------------------------------
Ran 9 tests in 0.096s

OK
```

## 课后习题

[第四章 决策树](./doc/decision_tree.md)  
[第五章 神经网络](./doc/neural_networks.md)  
