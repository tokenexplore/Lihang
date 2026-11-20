### 最大熵原理

最大熵原理是概率模型**学习的一个准则**，即学习概率模型时，在所有可能的概率模型（分布）中，熵最大的模型是最好的模型，通常带有约束条件。

所以，<span style='color:red'>最大熵原理是在满足约束条件的模型集合中选取熵最大的模型</span>

离散随机变量 $X$ 的概率分布是 $P(X)$，则其熵是
$$
H(P) = -\sum_xP(x)logP(x)
$$
满足不等式
$$
0 \le H(P) \le log|X|
$$
- $|X|$ 是 $X$ 的取值个数
- 当且仅当 $X$ 的分布是均匀分布是右边的等号成立，即均匀分布的熵最大,具体证明见[note](./note.md)

logistic 回归模型和最大熵模型，既可以看作是概率模型，又可以看作是非概率模型。

书中通过一个例子来介绍最大熵原理，下面引用一下文献中关于这个例子的总结。

> Model all that is known and assume nothing about that which is unknown. In other words, given a collection of facts, choose a model which is consistent with all the facts, but otherwise as uniform as possible.
>
> -- Berger, 1996

书中关于这部分的总结如下：**满足约束条件下求等概率的方法估计概率分布**

关于最大熵原理有很多直观容易理解的解释，比如Berger的例子，比如吴军老师数学之美中的例子。

最大熵原理很**常见**，很多原理我们都一直在用，只是没有上升到理论的高度。

- 最大熵原理认为要选择的概率模型首先必须满足**已有的事实**，即**约束条件**
- 最大熵原理根据已有的信息（**约束条件**），选择适当的概率模型。
- 最大熵原理认为不确定的部分都是等可能的，通过熵的最大化来表示**等可能性**，便于数值计算操作

<span style='color:red'>最大熵的原则，承认已有的，且对未知无偏。等概率表示了对事实的无知，因为没有更多的信息，这种判断是合理的。</span>

最大熵原理并不直接关心特征选择，但是特征选择是非常重要的，因为约束可能是成千上万的。

### 最大熵原理几何解释

这部分书中只描述了模型空间$\mathcal P$，两个约束$C_1$和$C_2$是**一致性**约束的情况。

在Berger 1996里面有展开这部分，分了四个图，分别讨论了

1. 概率模型空间$\mathcal {P}$
1. 单一约束$C_1$
1. 一致性(consistent)约束$C_1$和$C_2$，这种情况下模型唯一确定 $p=C_1\bigcap C_2$
1. 非一致性(inconsistent)约束$C_1$和$C_3$，这种情况下没有满足约束条件的模型。

##### 特征与约束条件

关于特征和约束，Berger有他的阐述

> 指示函数
>
> $f(x,y)=\begin{cases}
> 1 & if\ y=en\ and\ April\ follows\ in\\
> 0 & otherwise
> \end{cases}$

上面这个$f$直接引用自Berger的说明，原来的例子是英语in到法语的翻译。

这里面f就是**特征函数**，或者**特征**。

定义一个期望，如果是二值函数的话，就相当于计数。通过样本得到的这个统计。但是样本是有限的，并不是一个真实的分布， 所以叫经验分布，如果我们拿到的这个模型能够表示实际的分布，那么就可以假设经验分布和真实分布是相等的。这个，就是**约束方程**，或者**约束**。

一般模型的**特征**是关于$x$的函数，最大熵模型中的特征函数，是关于$x$和$y$的函数。注意理解$f(x)$与$f(x, y)$的区别。 关于特征函数可以参考[条件随机场](../CH11/README.md)中例11.1关于特征部分的内容增强理解。

### 模型定义

- 假设分类模型是一个条件概率分布，$P(Y|X)$, 输入 $X\in \mathcal {X} \sube \mathbf R^n，Y\in \mathcal {Y}$ 表示输出

- 给定一个训练集 $T=\{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$
$N$是训练样本容量，$x \in \mathbf R^n$ 
- 学习目标：应用最大熵原理选择最好的分类模型
- 联合分布$P(X,Y)$与边缘分布P(X)的经验分布分别为 $\widetilde P(X, Y)和 \widetilde P(X)$

$$
\begin{aligned}
&\widetilde P (X=x, Y=y)=\frac{\nu(X=x, Y=y)}{N} \\
&\widetilde P (X=x)=\frac {\nu (X=x)}{N}
\end{aligned}
$$
  - $\nu(X=x, Y=y)$ 表示样本(x,y)的频数
  - $\nu(X=x)$ 表示样本x的频数


上面两个就是不同的数据样本，在训练数据集中的比例。

如果增加$n$个**特征函数**, 就可以增加$n$个**约束条件**，特征也对应增加了一列。

假设满足所有约束条件的模型集合为

$\mathcal {C} \equiv \ \{P \in \mathcal {P}|E_P(f_i)=E_{\widetilde {P}}(f_i) {, i=1,2,\dots,n}\} $

定义在条件概率分布$P(Y|X)$上的条件熵为

$H(P)=-\sum \limits _{x, y} \widetilde {P}(x)P(y|x)\log {P(y|x)}$

则模型集合$\cal {C}$中条件熵$H(P)$最大的模型称为最大熵模型，上式中对数为自然对数。

特征函数$f(x,y)$关于经验分布$\widetilde P (X, Y)$的期望值用$E_{\widetilde P}(f)$表示

$$E_{\widetilde P}(f)=\sum\limits_{x,y}\widetilde P(x,y)f(x,y)$$

特征函数$f(x,y)$关于模型$P(Y|X)$与经验分布$\widetilde P (X)$的期望值, 用$E_{P}(f)$表示

$$E_{P}(f)=\sum\limits_{x,y}{\widetilde P(x)P(y|x)f(x,y)}$$

如果模型能够获取训练数据中的信息，那么就有
$$
\widetilde{P}(x,y)=P(y|x)\widetilde{P}(x)
$$
就可以假设这两个期望值相等，即

$$E_P(f)=E_{\widetilde P}(f)$$

上面这个也是约束方程

#### 算法实现

##### 特征提取原理

通过对已知训练集数据的分析，能够拿到联合分布的经验分布和边缘分布的经验分布。

特征函数 $f(x, y)$ 用来描述输入$x$和输出$y$之间的某一事实。
$$
f(x,y) = \begin{cases}
1 & x与y满足某一事实\\
0 & 否则
\end{cases}
$$

<span style='color:red'>这里，满足的事实，可以是in，显然，特征函数可以自己定义，可以定义多个，这些就是约束</span>

##### 预测分类原理

这里面重复一下书中的过程，在$L(P, w)$对$P$求导并令其为零的情况下解方程能拿到下面公式
$$
P(y|x)=\exp{\left(\sum_{i=1}^{n}w_if_i(x,y)+w_0-1\right)}=\frac{\exp{\left(\sum\limits_{i=1}^{n}w_if_i(x,y)\right)}}{\exp{\left(1-w_0\right)}}
$$
书中有提到因为$\sum\limits{_y}P(y|x)=1$，然后得到模型

$$
P_w(y|x)=\frac{1}{Z_w(x)}\exp{\sum\limits_{i=1}^{n}w_if_i(x,y)}\\
Z_w(x)=\sum_y\exp{\sum_{i=1}^{n}w_if_i(x,y)}
$$

注意这里面$Z_w$是归一化因子。

这里面并不是因为概率为1推导出了$Z_w$的表达式，而是因为$Z_w$的位置在分母，然后对应位置$\exp(1-w_0)$也在分母，凑出来这样一个表达式，意思就是遍历$y$的所有取值，求分子表达式的占比。

综上，如果$f_i(x,y)$只检测是不是存在这种组合，那么概率就是*归一化的*出现过的特征，系数求和再取e指数。

### 最大熵模型的学习

最大熵模型的学习过程就是求解最大熵模型的过程。

最大熵模型的学习可以形式化为约束最优化问题。

$$
\min \limits_{P\in \mathcal {C}}-H(P)=\sum\limits_{x,y}\widetilde P(x)P(y|x)\log P(y|x) \\
s.t. E_P(f_i)-E_{\widetilde P}(f_i)=0, i =1,2,\dots,n \\
\sum \limits_y P(y|x)=1
$$

可以通过例6.2 来理解最大熵模型学习的过程，例6.2 考虑了两种约束条件，这部分内容可以通过python符号推导实现，下面代码整理整个求解过程。

#### 例6.2

###### 一个约束条件

```python
from sympy import *

# 1 constrains
P1, P2, P3, P4, P5, w0, w1, w2 = symbols("P1, P2, P3, P4, P5, w0, w1, w2", real=True)
L = P1 * log(P1) + P2 * log(P2) + P3 * log(P3) + P4 * log(P4) + P5 * log(P5) \
    + w0 * (P1 + P2 + P3 + P4 + P5 - 1)
P1_e = (solve(diff(L, P1), P1))[0]
P2_e = (solve(diff(L, P2), P2))[0]
P3_e = (solve(diff(L, P3), P3))[0]
P4_e = (solve(diff(L, P4), P4))[0]
P5_e = (solve(diff(L, P5), P5))[0]
L = L.subs({P1: P1_e, P2: P2_e, P3: P3_e, P4: P4_e, P5: P5_e})
w = (solve([diff(L, w0)], [w0]))[0]
P = [P1_e.subs({w0: w[0]}),
     P2_e.subs({w0: w[0]}),
     P3_e.subs({w0: w[0]}),
     P4_e.subs({w0: w[0]}),
     P5_e.subs({w0: w[0]})]
P
```
##### 两个约束条件
```python
# 2 constrains
P1, P2, P3, P4, P5, w0, w1, w2 = symbols("P1, P2, P3, P4, P5, w0, w1, w2",real=True)
L = P1*log(P1) + P2*log(P2)+P3*log(P3)+P4*log(P4)+P5*log(P5)\
    +w1*(P1+P2-3/10)\
    +w0*(P1+P2+P3+P4+P5-1)
P1_e = (solve(diff(L,P1),P1))[0]
P2_e = (solve(diff(L,P2),P2))[0]
P3_e = (solve(diff(L,P3),P3))[0]
P4_e = (solve(diff(L,P4),P4))[0]
P5_e = (solve(diff(L,P5),P5))[0]
L = L.subs({P1:P1_e, P2:P2_e, P3:P3_e, P4:P4_e, P5:P5_e})
w = (solve([diff(L,w1),diff(L,w0)],[w0,w1]))[0]
P = [P1_e.subs({w0:w[0], w1:w[1]}),
     P2_e.subs({w0:w[0], w1:w[1]}),
     P3_e.subs({w0:w[0], w1:w[1]}),
     P4_e.subs({w0:w[0], w1:w[1]}),
     P5_e.subs({w0:w[0], w1:w[1]})]
P
```

###### 三个约束条件

```python
# 3 constrains
P1, P2, P3, P4, P5, w0, w1, w2 = symbols("P1, P2, P3, P4, P5, w0, w1, w2",real=True)
L = P1*log(P1) + P2*log(P2)+P3*log(P3)+P4*log(P4)+P5*log(P5)\
    +w2*(P1+P3-1/2)\
    +w1*(P1+P2-3/10)\
    +w0*(P1+P2+P3+P4+P5-1)
P1_e = (solve(diff(L,P1),P1))[0]
P2_e = (solve(diff(L,P2),P2))[0]
P3_e = (solve(diff(L,P3),P3))[0]
P4_e = (solve(diff(L,P4),P4))[0]
P5_e = (solve(diff(L,P5),P5))[0]
L = L.subs({P1:P1_e, P2:P2_e, P3:P3_e, P4:P4_e, P5:P5_e})
w = (solve([diff(L,w2),diff(L,w1),diff(L,w0)],[w0,w1,w2]))[0]
P = [P1_e.subs({w0:w[0], w1:w[1],w2:w[2]}),
     P2_e.subs({w0:w[0], w1:w[1],w2:w[2]}),
     P3_e.subs({w0:w[0], w1:w[1],w2:w[2]}),
     P4_e.subs({w0:w[0], w1:w[1],w2:w[2]}),
     P5_e.subs({w0:w[0], w1:w[1],w2:w[2]})]
P
```

## 模型学习

逻辑斯谛回归模型和最大熵模型学习归结为以**似然函数**为**目标函数**的最优化问题，通常通过迭代算法求解。

### 目标函数

#### 逻辑斯谛回归模型

$$
\begin{aligned}
L(w)&=\sum\limits^{N}_{i=1}[y_i\log\pi(x_i)+(1-y_i)\log(1-\pi(x_i))]\\
&=\sum\limits^{N}_{i=1}[y_i\log{\frac{\pi(x_i)}{1-\pi(x_i)}}+\log(1-\pi(x_i))]\\
&=\sum\limits^{N}_{i=1}[y_i(w\cdot x_i)-\log(1+\exp(w\cdot{x_i})]
\end{aligned}
$$

#### 最大熵模型

$$
\begin{align}
L_{\widetilde {P}}(P_w)&=\sum \limits_{x,y}\widetilde {P}(x,y)\log{P}(y|x)\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x,y}\widetilde{P}(x,y)\log{(Z_w(x))}\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x,y}\widetilde{P}(x)P(y|x)\log{(Z_w(x))}\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x}\widetilde{P}(x)\log{(Z_w(x))}\sum_{y}P(y|x)\\
&=\sum \limits_{x,y}\widetilde {P}(x,y)\sum \limits_{i=1}^{n}w_if_i(x,y) -\sum \limits_{x}\widetilde{P}(x)\log{(Z_w(x))}
\end{align}
$$

以上推导用到了$\sum\limits_yP(y|x)=1$

## 模型学习的最优化算法
logistic regression 模型和 最大熵模型学习归结为以似然函数为目标函数的最优化问题，通常使用迭代算法求解，常用的有
- 改进的迭代尺度法
- 梯度下降法
- 牛顿法或拟牛顿法：一般收敛速度更快

### 改进的迭代尺度法（IIS）

### 拟牛顿法
对于最大熵模型而言，

$$
P_w(y|x) = \frac{\exp\left(\sum_{i=1}^n w_i f_i(x, y)\right)}{\sum_y \exp\left(\sum_{i=1}^n w_i f_i(x, y)\right)}
$$

目标函数：

$$
\min_{w \in \mathbf{R}^n} f(w) = \sum_x \tilde{P}(x) \log \sum_y \exp\left(\sum_{i=1}^n w_i f_i(x, y)\right) - \sum_{x,y} \tilde{P}(x, y) \sum_{i=1}^n w_i f_i(x, y)
$$

梯度：

$$
g(w) = \left( \frac{\partial f(w)}{\partial w_1}, \frac{\partial f(w)}{\partial w_2}, \cdots, \frac{\partial f(w)}{\partial w_n} \right)^T
$$

其中

$$
\frac{\partial f(w)}{\partial w_i} = \sum_{x,y} \tilde{P}(x) P_w(y|x) f_i(x, y) - E_{\tilde{P}}(f_i), \quad i = 1, 2, \cdots, n
$$

相应的拟牛顿法 BFGS 算法如下。

**算法（最大熵模型学习的 BFGS 算法）**
输入：特征函数 $ f_1, f_2, \cdots, f_n $；经验分布 $ \tilde{P}(x, y) $，目标函数 $ f(w) $，梯度 $ g(w) = \nabla f(w) $，精度要求 $ \varepsilon $；

输出：最优参数值 $ w^* $；最优模型 $ P_{w^*}(y|x) $。

(1) 选定初始点 $ w^{(0)} $，取 $ B_0 $ 为正定对称矩阵，置 $ k = 0 $；

(2) 计算 $ g_k = g(w^{(k)}) $。若 $ \|g_k\| < \varepsilon $，则停止计算，得 $ w^* = w^{(k)} $；否则转 (3)；

(3) 由 $ B_k p_k = -g_k $ 求出 $ p_k $；
(4) 一维搜索: 求 $\lambda_k$ 使得

$$
f(w^{(k)} + \lambda_k p_k) = \min_{\lambda \geqslant 0} f(w^{(k)} + \lambda p_k)
$$

(5) 置 $w^{(k+1)} = w^{(k)} + \lambda_k p_k$;

(6) 计算 $g_{k+1} = g(w^{(k+1)})$, 若 $\|g_{k+1}\| < \varepsilon$, 则停止计算, 得 $w^* = w^{(k+1)}$; 否则, 按下式求出 $B_{k+1}$:

$$
B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T \delta_k} - \frac{B_k \delta_k \delta_k^T B_k}{\delta_k^T B_k \delta_k}
$$

其中,

$$
y_k = g_{k+1} - g_k, \quad \delta_k = w^{(k+1)} - w^{(k)}
$$

(7) 置 $k = k + 1$, 转 (3)。


### 其他

课后习题的第一个题目提到了**指数族**(Exponential family)分布，这个概念在PRML中有单独的章节进行阐述。

扩展一下指数族分布：

|          |      |      |      |      |      |
| -------- | ---- | ---- | ---- | ---- | ---- |
| 正态分布 |      |      |      |      |      |
| 两点分布 |      |      |      |      |      |
| 二项分布 |      |      |      |      |      |
| 泊松分布 |      |      |      |      |      |
| 伽马分布 |      |      |      |      |      |
|          |      |      |      |      |      |

大部分的算法实现，其实都是在刷权重，记得有个同学的昵称就是“一切源于wx+b”。另外，**其实算法能跑起来，能预测，并不能说明算法实现的正确的，或者说并不一定是最优的。** 但是其实这种情况也比较常见， 有的时候没有条件去创造一个专门的工具来解决问题的时候，或者没有更好的工具解决问题的时候， 我们会选择能解决部分问题，或者能解决问题的工具来**对付**

## 代码实现

关于代码实现，网上看似众多的版本，应该基本上都源自最早15年的一份GIS的程序。 

无论怎样，这些代码的实现，都会有助于对Maxent的理解。推荐后面参考文献[1]

李航老师在本章给出的参考文献中[1, 2]是Berger的文章。

### Demo

这部分代码没有LR的说明。

> 代码来源: https://vimsky.com/article/776.html
> 相关公式: https://vimsky.com/article/714.html

提几点:

1. 代码参考文献可以看berger的文章，公式编号基本对应。
1. 这份代码用defaultdict实现了稀疏存储。
1. 如果$f(x, y)$只是判断$(x, y)$在特征中出现的指示函数，那么特征可以简单的表示为$(x, y)$，这样给定一份含有标签的数据集，特征的数量就是 $m \times n$ 其中$m$是标签的数量，$n$是词表大小。注意书中注释过，$f(x,y)$可以是任意实值函数。
1. 这份代码思路很清晰，$(E_{\widetilde p}, Z_x \Rightarrow  P(y|x) \Rightarrow E_p) \Rightarrow \delta $，具体参考书中公式6.22，6.23，6.34
1. 体会一下在做直方图的时候，对于同一个样本，同样的特征出现多次的贡献是一样的。
1. 在未知的情况下，输出结果等概率。

### Maxent

>参考链接: https://github.com/WenDesi/lihang_book_algorithm/tree/master/maxENT

本来是想在这个代码的基础上更改，但是代码分解的不是非常容易理解。改来改去基本上面目全非了。保留链接以示感谢。博主写了一系列代码，至少有个成体系的参考。

提几点:

1. 没有用稀疏存储，所以，矩阵中会有很多零。需要除零错误处理

   ```python
   with np.errstate(divide='ignore', invalid='ignore'):
    tmp = np.true_divide(self.EPxy, self.EPx)
       tmp[tmp == np.inf] = 0
       tmp = np.nan_to_num(tmp)
   ```

   分子分母都是0对应nan，分母为0对应inf

1. 尝试了三种数据，可以通过命令行参数实现数据选择。

   - Demo中用到的data
   - train_binary.csv 这份数据的来源是参考链接中的，只考虑了0，1两种数据，标签少。
   - sklearn中的digits，标签全，但是8x8大小，比mnist少。其实8x8也不是一个非常小的图了，因为数字相对简单一点，用OpenCV做级联分类器的训练的时候，建议的图片大小是20x20，或者40x40，或者60x60不要太大

1. 书中有一个地方还是不理解的，提到了$f^\#$是不是常数的问题。

1. 没有采用字典的方式实现稀疏存储，但是numpy的数组操作还是很便捷的，后面有空评估一下存储和计算资源的消耗情况。

1. 大多数算法都是在刷权重，考虑哪些量(特征)可以用，哪些方法(算法)可以让权重刷的更合理，哪些方法(优化方法)能刷的更快。

### Mnist

有同学问LR实现中的GD，才发现那段代码不是很好读。而且，用到的train.csv已不在。

加了一个mnist_sample.py从Lecun那里下载数据，并按照类别采样300条。用来完成LR的Demo。

有些程序的问题，配合数据来理解。通常用到label乘法都是利用了label的符号，或者one-hot之后为了取到对应的类别的值。

代码更新了下，建议运行logistic_regression.py的时候在注释的位置断点，看下各个数据的shape，希望对理解代码有帮助。

## 参考

1. [Berger,1995, A Brief Maxent Tutorial](https://www.cs.cmu.edu/afs/cs/user/aberger/www/html/tutorial/tutorial.html)

1. [数学之美:信息的度量和作用]

1. [数学之美:不要把鸡蛋放在一个篮子里 谈谈最大熵模型]

1. [李航·统计学习方法笔记·第6章 logistic regression与最大熵模型（2）·最大熵模型](https://blog.csdn.net/tina_ttl/article/details/53542004)

1. [最大熵模型与GIS ,IIS算法](https://blog.csdn.net/u014688145/article/details/55003910)

1. [关于最大熵模型的严重困惑：为什么没有解析解？](https://www.zhihu.com/question/49139674/answer/114670380)

1. [最大熵模型介绍](http://www.cnblogs.com/hexinuaa/p/3353479.html) 这个是Berger的文章的翻译.

1. [理论简介](https://vimsky.com/article/714.html)  [代码实现](https://vimsky.com/article/776.html) 

1. [另外一份代码](https://github.com/WenDesi/lihang_book_algorithm/tree/master/maxENT)

1. [如何理解最大熵模型里面的特征？](https://www.zhihu.com/question/24094554)

1. [Iterative Scaling and Coordinate Descent Methods for
    Maximum Entropy Models](https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_journal.pdf)

1. [^1]: [Generative and discriminative classifiers: Naive Bayes and logistic regression](http://www.cs.cmu.edu/~tom/mlbook/NBayeslogReg.pdf)

1. [^2]: [On Discriminative vs. Generative Classifiers: A comparison of Logistic Regression and Naive Bayes](-)

1. [^3]: [ThinkBayes](http://www.greenteapress.com/thinkbayes/thinkbayes.pdf)

1. [^4]: [Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

1. https://blog.csdn.net/u012328159/article/details/72155874

**[⬆ top](#导读)**