---
layout: post
title: "逻辑回归LR推导（sigmoid，损失函数，梯度，参数更新公式）"
date: 2017-05-23 20:00:00
description: "逻辑回归LR推导（sigmoid，损失函数，梯度，参数更新公式）"
category: [machine learning]
tags: [math]
---

主要参考文献：[The equivalence of logistic regression and maximum entropy models,John Mount](http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf)
<!--more-->

## 一、声明
1. $ x(1),x(2),...,x(m) $ 表示 n 维空间的一个样本，$x(i)$ 表示第i个样本，$x(i)_j$ 表示第i个样本的第j维的数据（因为$x$是一个n维向量）。
2. $y(1),y(2),...,y(m)$ 表示 k 维空间的一个观测结果，记k从1,2,...,k变化，即分类问题中的k个类别，也可以0为下标开始，不影响推导。
3. $\pi()$是我们学习到的概率函数，实现样本数据到预测结果的映射：$R^n\rightarrow R^k$，（其实就是样本经过函数 $\pi()$计算后得到各个类别的预测概率，即一个k维向量），
$\pi(x)_u$表示数据样本x属于类别u的概率，我们希望$\pi()$具有如下性质：
> 1. $\pi(x)_v>0$  (样本x属于类别v的概率大于0，显然概率必须大于0)
> 2. $\sum_{v=1}^k\pi(x)_v = 1$,样本x属于各个类别的概率和为1
> 3. $\pi(x(i))_{y(i)}在所有类别概率中最大$

4.  $A(u,v)$是一个指示函数，$当u=v时A(u,v)=1，当u\neq v时A(u,v)=0，如A(u,y(i))$表示第i个观测结果是否为u

## 二、逻辑回归求解分类问题过程
对于二分类问题有k=2，对线性回归函数$\lambda x$进行非线性映射得到：
{% math %}\pi(x)_1 = \frac{\rm e^{\lambda \cdot x}}{\rm e^{\lambda \cdot x}+1}\tag{1}{% endmath %}
{% math %}\pi(x)_2 = 1-\pi(x)_1= \frac{1}{\rm e^{\lambda \cdot x}+1}\tag{2}{% endmath %}
对于多分类问题有：
{% math %}\pi(x) = \frac{\rm e^{\lambda _v\cdot x}}{\sum_{u=1}^m\rm e^{\lambda_u \cdot x}}\tag{3}{% endmath %}
对$\lambda$求偏导可得：
{% math %}
\begin{aligned}
u = v 时，
\frac {\partial\,\pi (x)_v}{\lambda_{v,j}} &= \frac{x_j\rm e^{\lambda _{v,j}\cdot x}\cdot \sum_{u=1}^m\rm e^{\lambda_{u,j} \cdot x}-x_j\rm e^{\lambda _{v,j}\cdot x}\rm e^{\lambda _{v,j}\cdot x}}{(\sum_{u=1}^m\rm e^{\lambda_{u,j} \cdot x})^2}\\
& = \frac{x_j\rm e^{\lambda _{v,j}\cdot x}}{\sum_{u=1}^m\rm e^{\lambda_{u,j} \cdot x}} \cdot \frac{\sum_{u=1}^m\rm e^{\lambda_{u,j} \cdot x}-\rm e^{\lambda_{v,j}\cdot x}}{\sum_{u=1}^m\rm e^{\lambda_{u,j} \cdot x}}\\
& = x_j \pi(x)_v(1-\pi(x)_v)
\end{aligned}
\tag{4}
{% endmath %}
{% math %}
\begin{aligned}
u \neq v 时
\frac {\partial\,\pi (x)_v}{\lambda_{u,j}}&=-\frac{\rm e^{\lambda_{v,j} \cdot x} \cdot (x_j\rm e^{\lambda_{u,j} \cdot x})}{(\sum_{u=1}^m\rm e^{\lambda_{u,j} \cdot x})^2} \\
&= -x_j \pi(x)_v\pi(x)_u, u\neq v时
\end{aligned}
\tag{5}
{% endmath %}
该分类问题的最大似然函数为：
{% math %}
L(\lambda)=\prod_{i=1}^m \pi(x(i))_{y(i)}\tag{6}
{% endmath %}
取对数得：
{% math %}
f(\lambda)=\log L(\lambda)=\sum_{i=1}^m \log(\pi(x(i))_{y(i)})\tag{7}
{% endmath %}
求似然函数最大值，令：
{% math %}
\begin{aligned}
\frac{\partial\,f(\lambda)}{\partial \,\lambda_{u,j}} &=\frac{\partial}{\partial \,\lambda_{u,j}}\sum_{i=1}^m \log(\pi(x(i))_{y(i)}) \\
&= \sum_{i=1}^m \frac{1}{\pi(x(i))_{y(i)}}\frac{\partial}{\partial \,\lambda_{u,j}}\pi(x(i))_{y(i)} \\
&= \sum_{\begin{array}{c}i=1,\\y(i)=u\end{array}}^m \frac{1}{\pi(x(i))_{y(i)}}\frac{\partial}{\partial \,\lambda_{u,j}}\pi(x(i))_{u} + \sum_{\begin{array}{c}i=1,\\y(i)\neq u\end{array}}^m \frac{1}{\pi(x(i))_{y(i)}}\frac{\partial}{\partial \,\lambda_{u,j}}\pi(x(i))_{y(i)}\\
&= \sum_{\begin{array}{c}i=1,\\y(i)=u\end{array}}^m \frac{1}{\pi(x(i))_{y(i)}}x(i)_j\pi(x(i))_u(1-\pi(x(i))_u) \\
&\quad - \sum_{\begin{array}{c}i=1,\\y(i)\neq u\end{array}}^m \frac{1}{\pi(x(i))_{y(i)}}x(i)_j\pi(x(i))_{y(i)} \pi(x(i))_u\\
&= \sum_{\begin{array}{c}i=1,\\y(i)=u\end{array}}^m x(i)_j(1-\pi(x(i))_u)-\sum_{\begin{array}{c}i=1,\\y(i)\neq u\end{array}}^m x(i)_j \pi(x(i))_u \\
&= \sum_{\begin{array}{c}i=1,\\y(i)=u\end{array}}^m x(i)_j - \sum_{i=1}^m x(i)_j\pi(x(i))_u \\
&= 0
\end{aligned}
\tag{8}
{% endmath %}
得：
{% math %}\sum_{\begin{array}{c}i=1,\\y(i)=u\end{array}}^m x(i)_j = \sum_{i=1}^m x(i)_j\pi(x(i))_u\tag{9}{% endmath %}
代入$A(u,y(i))=1$得：
{% math %}\sum_{i=1}^m x(i)_j\pi(x(i))_u = \sum_{i=1}^m x(i)_jA(u,y(i))\tag{10}{% endmath %}
综上有：
{% math %}
\frac{\partial\,f(\lambda)}{\partial \,\lambda_{u,j}}=\sum_{i=1}^m x(i)_j(A(u,y(i))-\pi(x(i))_u)
\tag{11}
{% endmath %}
则参数更新公式为：
{% math %}
\begin{aligned}
\lambda_{u,j} &= \lambda_{u,j} - \alpha \cdot \frac{\partial\,f(\lambda)}{\partial \,\lambda_{u,j}} \\
&= \lambda_{u,j} - \alpha \cdot \sum_{i=1}^m x(i)_j(A(u,y(i))-\pi(x(i))_u)
\end{aligned}
\tag{12}
{% endmath %}
### **那这就就存在个问题：为什么一开始要使用sigmoid函数进行非线性映射呢？其他函数不行吗？sigmoid函数怎么得来的？**

## 三、sigmoid函数的由来（最大熵）

由上文已知$\pi()$具应有如下性质：
> 1.  样本x属于类别v的概率大于0，显然概率必须大于0$\pi(x)_v>0\tag{13}$ 
> 2. 样本x属于各个类别的概率和为1 $\sum_{v=1}^k\pi(x)_v = 1\tag{14}$
> 3. $\pi(x(i))_{y(i)}在所有类别概率中最大$

其中对最后一个条件等价于尽可能的让$\pi(x(i))\rightarrow y(i)$ 即 $\pi(x(i))\rightarrow A(u,y(i))$，理想情况为$\pi(x(i))= A(u,y(i))$固有：
{% math %}\sum_{i=1}^m x(i)_j\pi(x(i))_u = \sum_{i=1}^m x(i)_jA(u,y(i))\tag{15}，对所有的u，j都成立{% endmath %}

对所有类别及所有样本取$\pi()$的熵，得：
{% math %}
f(v,i)=-\sum_{v=1}^k\sum_{i=1}^m\pi(x(i))_v \log(\pi(x(i))_v)
\tag{16}
{% endmath %}
得到一个优化问题：
{% math %}
\left\{
\begin{aligned}
max \, f(v,i)&=max \, (-\sum_{v=1}^k\sum_{i=1}^m\pi(x(i))_v \log(\pi(x(i))_v))\\
\pi(x)_v&>0\\
\sum_{v=1}^k\pi(x)_v &= 1\\
\sum_{i=1}^m x(i)_j\pi(x(i))_u &= \sum_{i=1}^m x(i)_jA(u,y(i))
\end{aligned}
\right.
\tag{17}
{% endmath %}
利用拉格朗日对偶性求这个优化问题的对偶问题，首先引入拉格朗日函数：
{% math %}
\begin{aligned}
L &= \sum_{j=1}^n\sum_{v=1}^k\lambda_{v,j} \left( \sum_{i=1}^m\pi(x(i))_vx(i)_j-A(v,y(i))x(i)_j \right)\\
&+ \sum_{v=1}^k\sum_{i=1}^m\beta_i(\pi(x(i))_v-1)\\
&- \sum_{v=1}^k\sum_{i=1}^m\pi(x(i))_v\log(\pi(x(i))_v)
\end{aligned}
\tag{18}
{% endmath %}
其中{% math %}\beta<0{% endmath %},由KKT条件有：
{% math %}
\frac{\partial\,L}{\partial \,\pi(x(i))_u} = \lambda_u\cdot x(i)+\beta_i-\log(\pi(x(i))_u) - 1 = 0  \quad对所有i,u \tag{19}
{% endmath %}
{% math %}
则：\pi(x(i))_u = e^{\lambda_u \cdot x(i)+\beta_i-1} \tag{20}
{% endmath %}
由（14）式得到：
{% math %}
 \sum_{v=1}^k e^{\lambda_u \cdot x(i)+\beta_i-1} = 1\\
即：e^\beta=\frac{1}{\sum_{v=1}^ke^{\lambda_u \cdot x(i)-1}} \tag{21}
{% endmath %}
代入（21）式消去常数项得：
{% math %}\pi(x(i))_u=\frac{e^{\lambda_u \cdot x}}{\sum_{v=1}^ke^{\lambda_u \cdot x}}\tag{22}{% endmath %}
即多分类问题对应的sigmoid函数
