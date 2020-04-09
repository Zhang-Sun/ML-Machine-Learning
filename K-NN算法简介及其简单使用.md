<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
# 1.K-临近算法

   本文简单介绍了K-临近算法的大致思想，并使用python3进行实战训练。

## 1.1 k-临近算法

  k-临近算法是一种基本的分类与回归算法。其工作原理是：存在一个训练样本，并且样本中每个数据都有与之对应的标签。当输入没有标签的数据后，将新的数据与样本集合中的数据对应的特征进行比较，然后提取样本中最相似的数据（最近邻）的分类标签。通常我们只选择样本中前k个最相似的数据，这就是k-近邻算法。k一般不大于20，最后选择前k个最相似护具中出现次数最多的分类标签作为新数据的标签。

如下例子是通过k-临近算法分类一个电影是爱情片还是动作片

<table>
    <tr>
        <td>电影名称</td>
        <td>打斗镜头</td>
        <td>接吻镜头</td>
        <td>电影类型</td>
     </tr>
    <tr>
        <td>电影1</td>
        <td>1</td>
        <td>101</td>
        <td>爱情片</td>
     </tr>
    <tr>
        <td>电影2</td>
        <td>5</td>
        <td>89</td>
        <td>爱情片</td>
    </tr>
    <tr>
        <td>电影3</td>
        <td>108</td>
        <td>5</td>
        <td>动作片</td>
    </tr>
    <tr>
        <td>电影4</td>
        <td>115</td>
        <td>8</td>
        <td>动作片</td>
    </tr>
</table>
                                                            
                                              表1.1 每部电影中的打斗镜头和接吻镜头以及电影类型
        

表1.1 即为我们的训练样本。这个数据集一共有两个特征即打斗镜头和接吻镜头，其标签即为爱情片和动作片。用我们肉眼看不难发现，打斗镜头多的是动作片，而接吻镜头多的是爱情片。倘若你再给我一个电影里的打斗镜头和接吻镜头数，我立刻就能判断出是动作片还是爱情片。而k-近邻算法也可以像我们一样，唯一的区别是它只能通过已有的数据集去判断新的电影属于哪一类。它会提取样本集中特征量最相似的数据分类标签作为新电影的标签。

## 1.2距离度量

   我们已经知道k-近邻算法根据特征比较，然后提取出样本集中特征最相似的数据的分类标签，那么它又是如何比较特征量是否相似呢？在这个例子中，我们的特征量就是打斗镜头和接吻镜头数量，因此，这个样本集就有2个特征量。倘若新给的数据为打斗镜头101，接吻镜头20，那么是距离哪些样本更近呢？这里我们给出距离计算公式：
$$ |AB| = \sqrt[2]{(x_1-x_2)^2+(y_1-y_2)^2} $$

通过计算我们可以得到如下结果
* (101,20)->动作片(108,5)的距离约为16.55
* (101,20)->动作片(115,8)的距离约为18.44
* (101,20)->爱情片(5,89)的距离约为118.22
* (101,20)->爱情片(1,101)的距离约为128.69

通过上述计算可知，新电影距离动作片（108，5）最近，为16.55。如果算法直接根据这个结果判断其为动作片，这个算法就是最近邻算法，而非k-近邻算法。k-近邻算法的步骤如下：
1. 计算已知类别的数据集中方的点雨当前点之间的距离。
2. 按照距离递增次序排序。
3. 选取与当前点距离最小的k个点。
4. 确定前k个点所在类别出现的频率。
5. 返回前k个点所出现频率最高的类别作为当前点的预测分类。

 比如，这里k取3，那么在电影例子中，按照距离由小到大排序三个点分别是动作片（108，5）、动作片（115，8）、爱情片（5，89）。在这三个点中，动作片出现的频率是三分之二，而爱情片出现的频率只有三分之一，所以新电影为动作片，这个判别过程就是K-临近算法。

## 1.3 Python代码实现

#### 1.3.1 准备数据集

对于表1.1中的数据，我们可以直接用numpy创建，代码如下：


```python
# -*- coding：UTF-8 -*-
import numpy as np

"""
函数说明：创建数据集

Parameter：
        none
Returns:
        group -数据集
        labels -分类标签
        
Modify:
        2020-4-9
"""

def createDataSet():
    #四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels

if __name__=="__main__":
    #创建数据集
    group , labels = createDataSet()
    #打印数据集
    print(group)
    print(labels)
```

    [[  1 101]
     [  5  89]
     [108   5]
     [115   8]]
    ['爱情片', '爱情片', '动作片', '动作片']


运行结果如下：
[[  1 101]
 [  5  89]
 [108   5]
 [115   8]]
['爱情片', '爱情片', '动作片', '动作片']

#### 1.3.2 k-近邻算法实现

根据两点距离公式，计算距离，选择距离最小的前k个点并返回分类结果。


```python
# -*- coding: UTF-8 -*-
import numpy as np
import operator


"""
函数说明：KNN算法，分类器

Parameter：
        inX - 用于分类的数据集
        dataSet - 用于测试的数据
        labels - 分类标签
        k - KNN算法参数，选择距离最小的k个点
        
Returns：
        sortedClassCount[0][0] -分类结果
        
Modify：
        2020-04-09
"""

def classify0(inX, dataSet, labels, k):
    #numpy函数的shape[0]返回dataSet的行数
    dataSetSize = dataSet,shape[0]
    #在列方向上重复inX共1次（横向），行向量方向上重复inX共dataSetSize次（纵向）
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat **2
    #sum()所有元素相加，sum(0)列相加，sum(1)行想加。
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，计算出距离
    distances = sqDistances**0.5
    #返回distances 中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get方法，返回指定键的值，如果值不在字典中返回默认值
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        #python3中的items()函数代替python2中的iteritems()函数
        #key=operator.itemgetter(1)根据字典的值进行排序
        #key = operator.itemgetter(0)根据字典的键进行排序
        #reverse降序排序字典
        sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
        #返回次数最多的类别，即所要分类的类别
        return sortedClassCount[0][0]
```

#### 1.3.3 整体代码


```python
# -*- coding: UTF-8 -*-
import numpy as np
import operator


"""
函数说明：创建数据集

Parameter：
        none
Returns:
        group -数据集
        labels -分类标签
        
Modify:
        2020-4-9
"""

def createDataSet():
    #四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels

"""
函数说明：KNN算法，分类器

Parameter：
        inX - 用于分类的数据集
        dataSet - 用于测试的数据
        labels - 分类标签
        k - KNN算法参数，选择距离最小的k个点
        
Returns：
        sortedClassCount[0][0] -分类结果
        
Modify：
        2020-04-09
"""

def classify0(inX, dataSet, labels, k):
    #numpy函数的shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列方向上重复inX共1次（横向），行向量方向上重复inX共dataSetSize次（纵向）
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat **2
    #sum()所有元素相加，sum(0)列相加，sum(1)行想加。
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，计算出距离
    distances = sqDistances**0.5
    #返回distances 中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get方法，返回指定键的值，如果值不在字典中返回默认值
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        #python3中的items()函数代替python2中的iteritems()函数
        #key=operator.itemgetter(1)根据字典的值进行排序
        #key = operator.itemgetter(0)根据字典的键进行排序
        #reverse降序排序字典
        sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
        #返回次数最多的类别，即所要分类的类别
        return sortedClassCount[0][0]
    
if __name__=='__main__':
    #创建数据集
    group , labels = createDataSet()
    #测试集
    test = [101,20]
    #KNN分类
    test_class = classify0(test, group, labels ,3)
    #打印分类结果
    print(test_class)
```

    动作片


我们所用到的例子中特征是2维的，这样的距离度量可以用二维平面上两点间的距离公式计算，但是如果是更高纬呢？此时我们可以使用欧式距离计算：$$ d(p,q) = d(q,p) = \sqrt[2]{(q_1-p_1)^2+(q_2-p_2)^2+......+(q_n-p_n)^2} = \sqrt[2]{\sum{(q_i-p_i)^2}}$$

kNN算法并不是一直都是正确的，我们可以使用多种方法检验分类器的正确率。为了测试分类器的效果，我们可以使用已知答案的数据。通过大量的测试数据，我们可以得到分类器的错误率-分类器给出错误结果的次数处以测试执行的总数。


```python

```
