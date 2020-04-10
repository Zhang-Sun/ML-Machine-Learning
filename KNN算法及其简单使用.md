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
   ![](http://latex.codecogs.com/gif.latex?|AB|=\\sqrt[2]{(x_1-x_2)^2+(y_1-y_2)^2} )

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


我们所用到的例子中特征是2维的，这样的距离度量可以用二维平面上两点间的距离公式计算，但是如果是更高纬呢？此时我们可以使用欧式距离计算：
![](http://latex.codecogs.com/gif.latex?d(p,q)=d(q,p)=\\sqrt[2]{(q_1-p_1)^2+(q_2-p_2)^2+......+(q_n-p_n)^2}=\sqrt[2]{\sum{(q_i-p_i)^2}})


kNN算法并不是一直都是正确的，我们可以使用多种方法检验分类器的正确率。为了测试分类器的效果，我们可以使用已知答案的数据。通过大量的测试数据，我们可以得到分类器的错误率-分类器给出错误结果的次数处以测试执行的总数。

# 2.k-近邻算法实战之约会网站配对效果判定

之前介绍的k-近邻算法实现方法并不是完整的k-近邻算法流程，完整的k-近邻算法的是流程一般是：
1. 收集数据：可使用爬虫或使用免费的数据网站，一般来讲，数据保存在txt格式文档中便于后续解析处理。
2. 准备数据：使用Pyhton解析、预处理数据。
3. 分析数据：可以使用很多方法对数据进行分析，例如使用matplotilb对数据进行可视化处理。
4. 测试算法：计算错误率。
5. 使用算法：错误率在可接受范围内史，就可以运行k-近邻算法进行分类。


以下是k-近邻算法的实战内容。

## 2.1 实战背景

海伦女士一直使用在线约会网站寻找自己的约会对象。经过一番总结后，她发现自己约会过的人可以进行如下分类：
* 不喜欢的人
* 魅力一般的人
* 极具魅力的人


海伦收集约会的数据已经有一段时间了，他把这些数据存放在文本文件datingTestSet.txt中，每个样本数据占据一行，总共有1000行。

海伦手机的样本数据主要包含以下3种特征：
* 每年获得的飞行常客里程数
* 玩视频游戏所消耗的时间百分比
* 每周消费的冰淇淋公升数


<table>
    <tr>
        <td>里程数</td>
        <td>时间百分比</td>
        <td>冰淇淋公升数</td>
        <td>态度</td>
     </tr>
    <tr>
        <td>40920</td>
        <td>8.326976</td>
        <td>0.953952</td>
        <td>largeDoses</td>
    </tr>
    <tr>
        <td>14488</td>
        <td>7.153469</td>
        <td>1.673904</td>
        <td>smallDoses</td>
    </tr>
     <tr>
        <td>26052</td>
        <td>1.441871</td>
        <td>0.805124</td>
        <td>didntLike</td>
    </tr>
</table>

                                                                          
                                                                          数据样例

上表是数据的前三行，态度分为：largeDoses（极具魅力）、smallDoses（魅力一般）、didntLIke（不喜欢）。

## 2.2 准备数据：数据解析

将上述特征数据输入到分类器前，必须将待处理的数据的格式改变为分类器可接受的格式。而分类器所接受的格式是什么呢？上小结我们知道，要将数据分类为两部分，即特征矩阵和对应的分类标签向量。在代码中使用file2matrix函数来处理数据的格式。


```python
# -*- coding：UTF-8  -*- 
import numpy as np
"""
函数说明：打开并解析数据文件、对数据进行分类：1代表不喜欢，2代表魅力一般，3代表极具魅力。

Parameters:
        filename -文件名
Returns：
        returnMat - 特征矩阵
        classLabelVector - 分类Label向量
    
Create：
        2020-4-10
"""

def file2matrix(filename):
    #打开文件
    with open(filename) as fr:
        #读取文件内容
        arrayOLines = fr.readlines()
        #得到文件的行数
        numberOfLines = len(arrayOLines)
        #返回的Numpy矩阵，解析完成的数据：numberOfLines行，3列
        returnMat = np.zeros((numberOfLines,3))
        #返回的分类标签向量
        classLabelVector = []
        #行的索引值
        index = 0
        for line in arrayOLines:
            #s.strip(rm),当rm为空时，默认删除空白符（包括‘\n’,'\r','\t',' '）
            line = line.strip()
            #使用s.split(str = "",num=string,cout(str))将字符串根据‘\t’分隔符进行切片
            listFromLine = line.split('\t')
            #将前三列提取出来放入特征矩阵中
            returnMat[index,:] = listFromLine[0:3]
            #根据文本中的标记的喜欢程度进行分类
            if listFromLine[-1] == 'didntLike':
                classLabelVector.append(1)
            elif listFromLine[-1] == 'smallDoses':
                classLabelVector.append(2)
            elif listFromLine[-1] == 'largeDoses':
                classLabelVector.append(3)
            index += 1
    return returnMat,classLabelVector

"""
函数说明：main函数测试数据处理

Parameters：
        None
Returns：
        None
        
Create：
        2020-4-10
"""

if __name__=="__main__":
    #打开的文件名
    filename = 'datingTestSet.txt'
    #打开并处理数据
    datingDataMat, datingDataLabels = file2matrix(filename)
    print(datingDataMat)
    print(datingDataLabels)
    
```

    [[4.0920000e+04 8.3269760e+00 9.5395200e-01]
     [1.4488000e+04 7.1534690e+00 1.6739040e+00]
     [2.6052000e+04 1.4418710e+00 8.0512400e-01]
     ...
     [2.6575000e+04 1.0650102e+01 8.6662700e-01]
     [4.8111000e+04 9.1345280e+00 7.2804500e-01]
     [4.3757000e+04 7.8826010e+00 1.3324460e+00]]
    [3, 2, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 2, 1, 2, 3, 2, 3, 2, 3, 2, 1, 3, 1, 3, 1, 2, 1, 1, 2, 3, 3, 1, 2, 3, 3, 3, 1, 1, 1, 1, 2, 2, 1, 3, 2, 2, 2, 2, 3, 1, 2, 1, 2, 2, 2, 2, 2, 3, 2, 3, 1, 2, 3, 2, 2, 1, 3, 1, 1, 3, 3, 1, 2, 3, 1, 3, 1, 2, 2, 1, 1, 3, 3, 1, 2, 1, 3, 3, 2, 1, 1, 3, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 2, 1, 3, 1, 2, 1, 1, 2, 3, 2, 3, 2, 3, 2, 1, 3, 3, 3, 1, 3, 2, 2, 3, 1, 3, 3, 3, 1, 3, 1, 1, 3, 3, 2, 3, 3, 1, 2, 3, 2, 2, 3, 3, 3, 1, 2, 2, 1, 1, 3, 2, 3, 3, 1, 2, 1, 3, 1, 2, 3, 2, 3, 1, 1, 1, 3, 2, 3, 1, 3, 2, 1, 3, 2, 2, 3, 2, 3, 2, 1, 1, 3, 1, 3, 2, 2, 2, 3, 2, 2, 1, 2, 2, 3, 1, 3, 3, 2, 1, 1, 1, 2, 1, 3, 3, 3, 3, 2, 1, 1, 1, 2, 3, 2, 1, 3, 1, 3, 2, 2, 3, 1, 3, 1, 1, 2, 1, 2, 2, 1, 3, 1, 3, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 3, 2, 2, 3, 1, 2, 1, 1, 1, 3, 3, 2, 1, 1, 1, 2, 2, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 3, 2, 3, 3, 3, 3, 1, 2, 3, 1, 1, 1, 3, 1, 3, 2, 2, 1, 3, 1, 3, 2, 2, 1, 2, 2, 3, 1, 3, 2, 1, 1, 3, 3, 2, 3, 3, 2, 3, 1, 3, 1, 3, 3, 1, 3, 2, 1, 3, 1, 3, 2, 1, 2, 2, 1, 3, 1, 1, 3, 3, 2, 2, 3, 1, 2, 3, 3, 2, 2, 1, 1, 1, 1, 3, 2, 1, 1, 3, 2, 1, 1, 3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 1, 3, 2, 2, 1, 2, 1, 3, 2, 1, 3, 2, 1, 3, 1, 1, 3, 3, 3, 3, 2, 1, 1, 2, 1, 3, 3, 2, 1, 2, 3, 2, 1, 2, 2, 2, 1, 1, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 3, 1, 1, 2, 2, 1, 2, 2, 2, 3, 1, 1, 1, 3, 1, 3, 1, 3, 3, 1, 1, 1, 3, 2, 3, 3, 2, 2, 1, 1, 1, 2, 1, 2, 2, 3, 3, 3, 1, 1, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 1, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 2, 3, 2, 1, 2, 2, 2, 3, 2, 1, 3, 2, 3, 2, 3, 2, 1, 1, 2, 3, 1, 3, 3, 3, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 1, 3, 3, 2, 2, 2, 3, 1, 2, 1, 1, 3, 2, 3, 2, 3, 2, 3, 3, 2, 2, 1, 3, 1, 2, 1, 3, 1, 1, 1, 3, 1, 1, 3, 3, 2, 2, 1, 3, 1, 1, 3, 2, 3, 1, 1, 3, 1, 3, 3, 1, 2, 3, 1, 3, 1, 1, 2, 1, 3, 1, 1, 1, 1, 2, 1, 3, 1, 2, 1, 3, 1, 3, 1, 1, 2, 2, 2, 3, 2, 2, 1, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 1, 3, 2, 3, 2, 1, 2, 1, 1, 1, 2, 3, 2, 2, 1, 2, 2, 1, 3, 1, 3, 3, 3, 2, 2, 3, 3, 1, 2, 2, 2, 3, 1, 2, 1, 3, 1, 2, 3, 1, 1, 1, 2, 2, 3, 1, 3, 1, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 2, 2, 3, 1, 3, 1, 2, 3, 2, 2, 3, 1, 2, 3, 2, 3, 1, 2, 2, 3, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 3, 2, 1, 3, 3, 3, 1, 1, 3, 1, 2, 3, 3, 2, 2, 2, 1, 2, 3, 2, 2, 3, 2, 2, 2, 3, 3, 2, 1, 3, 2, 1, 3, 3, 1, 2, 3, 2, 1, 3, 3, 3, 1, 2, 2, 2, 3, 2, 3, 3, 1, 2, 1, 1, 2, 1, 3, 1, 2, 2, 1, 3, 2, 1, 3, 3, 2, 2, 2, 1, 2, 2, 1, 3, 1, 3, 1, 3, 3, 1, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 3, 2, 2, 1, 3, 1, 2, 3, 1, 3, 1, 3, 1, 1, 3, 2, 3, 1, 1, 3, 3, 3, 3, 1, 3, 2, 2, 1, 1, 3, 3, 2, 2, 2, 1, 2, 1, 2, 1, 3, 2, 1, 2, 2, 3, 1, 2, 2, 2, 3, 2, 1, 2, 1, 2, 3, 3, 2, 3, 1, 1, 3, 3, 1, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 1, 1, 3, 2, 1, 2, 1, 2, 2, 3, 2, 2, 2, 3, 1, 2, 1, 2, 2, 1, 1, 2, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 3, 2, 3, 3, 2, 2, 1, 1, 1, 3, 3, 1, 1, 1, 3, 3, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 3, 1, 1, 2, 3, 2, 2, 1, 3, 1, 2, 3, 1, 2, 2, 2, 2, 3, 2, 3, 3, 1, 2, 1, 2, 3, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 3, 3, 3]


可以看到我们的数据已经处理成功，接下来我们通过直观的图形化方式来观察数据。

## 2.3分析数据：数据可视化

编写showdatas函数，将数据可视化。


```python
%matplotlib inline
# -*- coding：UTF-8 -*-

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

"""
函数说明：打开并解析数据文件、对数据进行分类：1代表不喜欢，2代表魅力一般，3代表极具魅力。

Parameters:
        filename -文件名
Returns：
        returnMat - 特征矩阵
        classLabelVector - 分类Label向量
    
Create：
        2020-4-10
"""

def file2matrix(filename):
    #打开文件
    with open(filename) as fr:
        #读取文件内容
        arrayOLines = fr.readlines()
        #得到文件的行数
        numberOfLines = len(arrayOLines)
        #返回的Numpy矩阵，解析完成的数据：numberOfLines行，3列
        returnMat = np.zeros((numberOfLines,3))
        #返回的分类标签向量
        classLabelVector = []
        #行的索引值
        index = 0
        for line in arrayOLines:
            #s.strip(rm),当rm为空时，默认删除空白符（包括‘\n’,'\r','\t',' '）
            line = line.strip()
            #使用s.split(str = "",num=string,cout(str))将字符串根据‘\t’分隔符进行切片
            listFromLine = line.split('\t')
            #将前三列提取出来放入特征矩阵中
            returnMat[index,:] = listFromLine[0:3]
            #根据文本中的标记的喜欢程度进行分类
            if listFromLine[-1] == 'didntLike':
                classLabelVector.append(1)
            elif listFromLine[-1] == 'smallDoses':
                classLabelVector.append(2)
            elif listFromLine[-1] == 'largeDoses':
                classLabelVector.append(3)
            index += 1
    return returnMat,classLabelVector

"""
函数说明：可视化数据

Parameters：
        datingDataMat - 特征矩阵
        datingLabels - 分类Label

Returns：
        None
        
Create：
        2020-4-10
        
"""

def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    #本文是在mac电脑下，windows电脑路径换成：c:\windows\fonts\simsun.ttc
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc',size=14)
    #将fig画布分隔成1行1列，不共享x轴和y轴，fig画布大小为（13，8）
    #当nrow = 2，ncols=2时，代表fig画布被分成了4个区域，axs[0][0]代表第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13,8))
    
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        elif i == 2:
            LabelsColors.append('orange')
        elif i == 3:
            LabelsColors.append('red')
    #画出散点图，以datingDataMat矩阵的第一列（飞行常客历程），第二列（玩游戏时间比例）数据画散点数据，散点大小为15，透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=0.5)
    #设置标题、x轴label，y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗的时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩游戏所消耗时间占比',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold',color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold',color='black')
    
    #画出散点图，以datingDataMat矩阵的第一列（飞行常客里程），第三列（冰淇凌）数据画散点数据，三点大小为15，透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0],y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=0.5)
    #设置标题，x轴label，y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰淇淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold',color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold',color='black')
    
    #画出散点图，以datingDataMat矩阵的第2列（玩游戏），第三列（冰淇凌）数据画散点数据，三点大小为15，透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1],y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=0.5)
    #设置标题，x轴label，y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗的时间占比与每周消费的冰淇淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗的时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰淇淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold',color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold',color='black')
    
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    
    #添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses,largeDoses])
    
    #显示图片
    plt.show()
    
"""
函数说明：main函数

Parameters：
        None
Returns：
        None
Modify：
        2020-4-10
"""
if __name__=='__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    showdatas(datingDataMat, datingLabels)

    
```


![png](output_40_0.png)


通过图像可以清晰的看出来数据的规律。

## 2.4 准备数据：数据归一化

下表给出了四组样本，如果想要计算样本3和样本4之间的距离，可以使用欧拉公式计算。

<table>
    <tr>
        <td>样本</td>
        <td>玩游戏所消耗的时间百分比</td>
        <td>每年获得的飞行常用里程数</td>
        <td>每周消费的冰淇淋公升数</td>
        <td>样本分类</td>
     </tr>
    <tr>
        <td>1</td>
        <td>0.8</td>
        <td>400</td>
        <td>0.5</td>
        <td>1</td>
    </tr>
    <tr>
        <td>2</td>
        <td>12</td>
        <td>134000</td>
        <td>0.9</td>
        <td>3</td>
    </tr>
    <tr>
        <td>3</td>
        <td>0</td>
        <td>20000</td>
        <td>1.1</td>
        <td>2</td>
    </tr>
    <tr>
        <td>4</td>
        <td>67</td>
        <td>32000</td>
        <td>0.1</td>
        <td>2</td>
      
</table>

计算方法如下图所示：

 ![](http://latex.codecogs.com/gif.latex?\\sqrt[2]{(0-67)^2+(20000-32000)^2+(1.1-0.1)^2} )


我们发现，上面方程中对计算结果影响最大的是每年获得的飞行常用里程数，仅仅是因为它的数值比较大，而海伦认为每个特征的影响应该是相同的，因此我们需要给每个特征一个权重，来使三个特征的影响结果相同。

在处理这种不同取值范围的特征值时，我们通常采用的方法就是将数值归一化，将取值范围统一到0到1或者-1到1之间。下面的公式可以将任意取值范围的特征值转化到0到1之间：$$ newValue = (oldValue-min)/(max-min) $$

其中min和max分别是数据集中的最大值和最小值，虽然改变数值范围增加了分类器复杂度，但为了得到更准确的结果，我们必须这样做。编写autoNorm函数实现数据的归一化。


```python
# -*- coding: UTF-8 -*-
import numpy as np

"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量

Modify:
    2020-04-10
"""
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    #得到文件行数
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

"""
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

Modify:
    2020-04-10
"""
def autoNorm(dataSet):
    #获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals

"""
函数说明:main函数

Parameters:
    无
Returns:
    无

Modify:
    2020-4-10
"""
if __name__ == '__main__':
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    print(normDataSet)
    print(ranges)
    print(minVals)

```

    [[0.44832535 0.39805139 0.56233353]
     [0.15873259 0.34195467 0.98724416]
     [0.28542943 0.06892523 0.47449629]
     ...
     [0.29115949 0.50910294 0.51079493]
     [0.52711097 0.43665451 0.4290048 ]
     [0.47940793 0.3768091  0.78571804]]
    [9.1273000e+04 2.0919349e+01 1.6943610e+00]
    [0.       0.       0.001156]




```python

```
