# <div align="center"> Topic Model （主题模型）</div>  
***
<center>![主题模型](https://pic1.zhimg.com/v2-db1cb9e08d326d231d02ba3efa4ce0ff_1200x500.jpg "主题模型")</center>


## 引言
主题检测对于许多任务是一项非常有价值的。比如：文档聚类，特征选择，信息检索和推荐。
许多内容提供商和新闻代理使用主题模型来推荐文章给读者。通常来说，收集的原始数据集是非结构的。
所以我们需要强大的工具和技术来分析和理解大规模的非结构化数据。

### 主题模型
主题模型是处理非结构化数据的一种常用方法，是一个无监督的文本挖掘技术，旨在自动发现隐藏在文档背后的主题。主题可以由语料库中的共现词项所定义，它通过确定共现词的方法来总结大量的文本信息。它有助于发现文档背后隐藏的主题、标注文档和组织大规模的无结构化数据。主题模型假设每篇文档不归属于某一类，而是依据一定的概率分布在隐含主题上的。同时每个主题包含多个词。

主题模型在机器学习和自然语言处理等领域是用来在一系列文档中发现抽象主题的一种统计模型。直观来讲，如果一篇文章有一个中心思想，那么一些特定词语会更频繁的出现。比方说，如果一篇文章是在讲狗的，那“狗”和“骨头”等词出现的频率会高些。如果一篇文章是在讲猫的，那“猫”和“鱼”等词出现的频率会高些。而有些词例如“这个”、“和”大概在两篇文章中出现的频率会大致相等。但真实的情况是，一篇文章通常包含多种主题，而且每个主题所占比例各不相同。因此，如果一篇文章10%和猫有关，90%和狗有关，那么和狗相关的关键字出现的次数大概会是和猫相关的关键字出现次数的9倍。一个主题模型试图用数学框架来体现文档的这种特点。主题模型自动分析每个文档，统计文档内的词语，根据统计的信息来断定当前文档含有哪些主题，以及每个主题所占的比例各为多少。

### 对比主题模型和文本分类
文本分类是一个监督的机器学习问题，每篇文档被分类为预定义的一系列label中的一个。
主题模型是一个无监督学习形式，发现文档中共现词的聚类。这些聚类在一起的共现词称为“topic”。这些topic是未知的。主题模型能用来解决文本分类问题，呈现文档中主题的分布。
![avatar](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1538411402/image2_ndnai9.png "主题模型和文本分类的对比")


***
## LSA（Latent Semantic Analysis）
隐语义分析（Latent Semantic Analysis, LSA）又名隐语义索引（Latent Semantic Index, LSI）。LSA最初用在语义检索上，目的是为了找出词在文档和查询中的真正含义，也就是隐含语义，从而解决一词多义和一义多词的问题。LSA和传统的向量空间模型一样使用向量来表示词和文档，并通过向量之间的关系来判断词之间以及文档之间的关系。不同的是，传统的向量空间模型无法使用精确的词匹配，即精确地匹配用户输入的词和向量空间中存在的词，所以无法解决一词多义以及一义多词的问题。因为在**实际匹配中，我们想要比较的不是词，而是隐藏在词后面的意义和概念**。而**LSA将词和文档从高维空间映射到低维的语义空间，在比较其相似性**。从而**成功解决一义多词的问题，同时去除了原始向量空间中的一些噪声，提高特征的鲁棒性**。  

在基于单词的检索方法中，同义词（一义多词）会降低检索算法的召回率（recall），多义词（一词多义）会降低检索算法的精确率（precision）。

技术上，LSA使用词袋模型（bag of word, BoW）来表示原始“词－文档”的特征矩阵X；然后使用奇异值分解（Singular Value Decomposition, SVD）技术将特征矩阵Ｆ分解为“词－主题”矩阵U和“文档-主题”矩阵V；再通过选择奇异值中最大的t个数，且只保留Ｕ和Ｖ矩阵的前t列来　实现降维，从而达到过滤噪声和冗余数据的目的。
![avatar](https://seanlee97.github.io/images/posts/svd/lsa.png "LSA的SVD分解")

在查询时，对于每个给定的查询，我们根据这个查询中包含的单词(x)构造一个伪文:`$V_q=X_qUD^{-1}$`，然后将该伪文档跟右奇异矩阵V的每一行计算余弦相似度来得到和给定查询最相似的文档。  

矩阵含义：  
N: 词总数  
M: 文档总数  
K: 主题数  
X: 原始“词－文档”矩阵，使用词袋模型。($X \in \mathbb{R}^{N\times M}$)  
U: 左奇异矩阵Ｕ的每一列表示语义（主题）相关的词类。（$U \in \mathbb{R}^{N\times K}$）  
D: 对角矩阵Ｄ表示词类和文章类之间的相关性，非负。 （$D \in \mathbb{R}^{K\times K}$）  
V: 右奇异矩阵Ｖ的每一列表示语义（主题）相关的文档类。　($V\in \mathbb{R}^{M\times K}$)

**如何确定主题数Ｋ：**  
1. 将每个主题当做一个聚类，然后根据聚类评价标准（比如轮廓稀疏 Silhouette coefficient）来评价聚类好坏程度。  
2. 根据主题一致性（Topic Coherence）来度量每个主题的好坏。主题一致性是指使用主题内所有pairwise词之间的相似度的平均值/中位数来度量主题的好坏。


**优点：**  
1. 简单易实现，只包含文档－词矩阵的分解
2. 一义多词：　通过SVD分解映射到低维语义空间后，多个语义相似但不同的词，在低维语义空间也相似。  
3. 去除噪声：　通过SVD分解取t个奇异值来降维后，达到过滤噪声和冗余数据的目的。  
4. 无监督、自动化。  
5. 语言无关。  
**缺点：**  
1. SVD的优化目标基于L2-norm或者Frobenius Norm的，这相当与隐含了对数据的高斯分布假设。而词出现的次数是非负的，明显不符合高斯假设，而更接近Multi-nomial分布。  
2. 特征向量没有对应的物理解释。  
3. SVD计算复杂度高，而且当新文档来的时候需要重新训练模型。  
4. 没有刻画词出现次数的概率模型。  
5. 对于计数向量而言，欧式距离不合适（重建时会产生负数）。  
6. 隐含主题的数量依赖于矩阵的秩。 
7. 一词多义：　无法解决一词多义问题。映射到低维隐空间中的某个点，意思就固定了。  
8. 相比LDA较难实现，准确率也比LDA差。  




***
## PLSA (Probabilistic Latent Semantic Analysi)
概率隐语义分析（Probabilistic Latent Semantic Analysis, PLSA）基于LSA，使用概率模型，具有更明确的物理意义。PLSA在文档和词之间引入一个隐变量（topic，即文档主题），然后使用 期望最大化算法来学习模型参数。PLSA建模思想简单，针对观察到的变量使用似然函数建模，建模过程中暴露隐含变量，难以使用极大似然估计，所以使用EM算法求解。  

PLSA 可以同时解决一义多词和一词多义的问题。  

### 概率图模型
![PLSA](https://github.com/linjinzhong/Picture/blob/master/plsa.png?raw=true "PLSA概率图模型")  

其中D表示文档，Z表示主题，W表示词，箭头表示了变量间的依赖关系。PLSA引入隐含变量Z，认为{D,Z,W}表示完整的数据集（the complete data set），而原始真实数据集{D,W}是不完整数据集（the incomplete data）。

当原始数据的似然函数很复杂时，我们通过增加一些隐含变量来增强我们的数据，得到”complete data”，而“complete　data”的似然函数更加简单，方便求极大值。于是，原始的数据就成了“incomplete data”。我们可以通过最大化“complete　data”似然函数的期望来最大化"incomplete　data"的似然函数，以便得到求似然函数最大值更为简单的计算途径。  

假设Z的取值有K个。PLSA模型假设文档生成过程如下：  
1. 以$p(d_i)$的概率选择一个文档$d_i$  
2. 以$p(z_k|d_i)$的概率选择一个主题$z_k$    
3. 以$p(w_j|z_k)$的概率生成一个单词$w_j$  

根据图模型，可以得到观测数据的联合概率分布：  
$$\begin{align}
p(d_i,w_j)
 &= \sum_{k=1}^{K}p(d_i,w_j,z_k)\notag \\  
 &= \sum_{k=1}^{K}p(d_i)p(z_k|d_i)p(w_j|z_k)\notag\\
 &=p(d_i)\sum_{k=1}^{K}p(z_k|d_i)p(w_j|z_k)
\end{align}$$  
第一个等式是对三者的联合概率分布对其中的隐藏变量 Z 的所有取值累加，第二个等式根据图模型的依赖关系将联合概率展开为条件概率，第三个等式只是简单的乘法结合律。这样就计算出了第 i 篇文档与第 j 个单词的联合概率分布。

### 完整数据时的对数似然函数（complete-data log likelihood）  
$$\begin{align}
L
&= \sum_{i=1}^{N}\sum_{j=1}^{M}n(d_i,w_j)log(p(d_i,w_j))\notag\\
&= \sum_{i=1}^{N}\sum_{j=1}^{M}n(d_i,w_j)log(p(d_i)\sum_{k=1}^{K}p(z_k|d_i)p(w_j|z_k))\notag\\
&= \sum_{i=1}^{N}\sum_{j=1}^{M}n(d_i,w_j)log(p(d_i))+ \sum_{i=1}^{N}\sum_{j=1}^{M}n(d_i,w_j)log(\sum_{k=1}^{K}p(z_k|d_i)p(w_j|z_k))
\end{align}$$  
其中$n(d_i,w_j)$表示第j个词在第i个文档中出现的次数。等式右侧第一项对于给定的数据集来说是定值，我们只需要使得第二项取到最大。$p(z_k|d_i)$ 和 $p(w_j|z_k)$是PLSA模型需要求解的参数，按照通常的方法，可以通过对参数求导使用梯度下降或牛顿法等方法来求解，但是这里的参数以求和的形式出现在了对数函数之中，求导结果十分复杂，无法使用那些方法。可以使用EM算法。

### 期望最大化算法（Expectation Maximizationm, EM）
我们没有办法直接优化似然性L，而是通过找到一个辅助函数Q，使得每次增大辅助函数Q的时候确保L也能得到增加，图示如下：
![EM](https://github.com/linjinzhong/Picture/blob/master/EM.png?raw=true "EM算法优化")  

PLSA中的期望：  
$E=\sum_{i=1}^{N}\sum_{j=1}^{M}n(d_i,w_j)\sum_{k=1}^{K}p(z_k|d_i,w_j)log[p(z_k|d_i)p(w_j|z_k)]$

约束条件：  
$\sum_{j=1}^{M}p(w_j|z_k) = 1$  
$\sum_{k=1}^{K}p(z_k|d_i) = 1$  

应用拉格朗日乘数法，可以求得 

1. E-step（根据初始化的$p(z_k|d_i)$和$p(w_j|z_k)$来计算期望）：  
$p(z_k|d_i,w_j)=\frac{p(z_k|d_i)p(w_j|z_k)}{\sum_{k=1}{K}p(z_k|d_i)p(w_j|z_k)}$  
2. M-step:（根据更新的期望$p(z_k|d_i,w_j)$来更新$p(z_k|d_i)$和$p(w_j|z_k)$）：  
$p(w_j|z_k)=\frac{\sum_{i=1}^{N}n(d_i,w_j)p(z_k|d_i,w_j)}{\sum_{m=1}^{M}\sum_{i=1}^{N}n(d_i,w_j)p(z_k|d_i,w_j)}$  
$p(z_k|d_i)=\frac{\sum_{j=1}^{M}n(d_i,w_j)p(z_k|d_i,w_j)}{n(d_i)}$  



***
## LDA (Latent Dirichlet Allocation)

隐狄利克雷分布（Latent Dirichlet Allocation, LDA）是PLSA的泛化版本。PLSA定义一个概率图模型，假设数据的生成过程，但是不是一个完全的生成过程：没有给出先验，给出的是一个极大似然估计或者最大后验估计。而LDA将PLSA中的参数变成随机变量，并且加入狄利克雷先验得到贝叶斯模型。使用狄利克雷先验主要是利用了狄利克雷分布和多项式分布的共轭性，方便计算。当将LDA的超参数设为特定值时，就特化成PLSA。LDA和PLSA的本质区别是估计参数的思想不同，PLSA使用频率派的思想，LDA使用贝叶斯派的思想。

### 概率图模型
LDA概率图模型如下。其中α和β是两个不同的狄利克雷分布的参数，别用来生成隐含主题的分布参数$\theta$和词的分布参数$\varphi$。$z$和$w$分别是从各自分布中选出的隐含主题和特定的单词。LDA 模型假设文档是由一系列主题构成的，然后再从这些主题中依据相应的概率分布生成词语。给定一个文档数据集，LDA 模型主要用于识别文档中的主题分布情况。

![LDA](https://github.com/linjinzhong/Picture/blob/master/lda.png?raw=true)



***
### Reference:
[参考资料1](https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python#comments)  
[参考资料2](https://www.cnblogs.com/kemaswill/archive/2013/04/17/3022100.html)  
[参考资料3](https://zhuanlan.zhihu.com/p/23034092)  
[参考资料4](https://blog.csdn.net/KIDGIN7439/article/details/69831490)  
[参考资料5](http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/)  
[参考资料6](http://zhikaizhang.cn/2016/06/17/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B9%8BPLSA/)  
