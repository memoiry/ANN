# flann_lsh
Benchmark for p-stable local sensitive hash and kdtree method in flann.

A matlab interface is implemented.

## Usage

### Experiment of pyflann-kdtree and p-stable LSH

Install Pyflann, Seaborn, and download the source code from github.

```bash
pip install pyflann
pip install seaborn
git clone https://github.com/memoiry/flann_lsh
```

Put the sift and gist data in the corresponding data folder and run the command below.

```bash
cd flann_lsh/src
python run_exp.py
```

To generate the result figure, run the command below. The analysis will be put in figure folder.

```bash
python analysis.py
```

### PLSH class usage


PLSH is a class used to create a local sensitive hash object.

* PLSH(key_size, table_num, w)

When the lsh object is constructed, simply build the index using the training dataset.

* lsh.build_index(dataset)

Last, search the ann result for query dataset.

* lsh.query(testset, k)

```python
from tester import *
from utils import *

dataset = np.random.randn(10000,100)*100
testset = np.random.randn(100,100)*10000
w = 4
k_num = 5
k_set = [6, 12]
L_set = [12, 16, 20]
ground_truth = linear_search(dataset, testset, k_num)
print "ground truth computing finished"
print "k L    recall    touched    recall/touched"
for k in k_set:
	for L in L_set:
		lsh = PLSH(k, L, 5000)
		lsh.build_index(dataset)
		result, dists, candi_avg = lsh.query(testset, k_num)
		acc = recall(result, ground_truth, k_num)
		candi_avg = candi_avg * 100
		prop = acc / candi_avg
		print k, L, ' ',' ','{} %   {} %'.format(acc,candi_avg), ' ', ' ', prop
```


### Matlab Interface 

#### Example


lsh_search is a simple wrapper function used to call Python PLSH implementation from command line for MATLAB. 

* lsh_search(dataset,query,k,table_num,key_size)

```matlab
dataset = randn(10000,100);
query = randn(100,100);
k = 10;
table_num = 8;
key_size = 12;
[ground_truth, ground_truth_dists] = knnsearch(dataset,Y,'K',k);
result, dists = lsh_search(dataset,query,k,table_num,key_size);
acc = recall(result, ground_truth, k)
```

## Result


### Hardware 

* Macbook Pro 2014 mid 
* CPU: Intel Core i5, 3.30GHz, 8GB RAM

### SIFT1M

<p align="center">
    <img src="https://ooo.0o0.ooo/2017/04/10/58eb9f1e6b466.png" width="640">
</p>
<p align="center" style="color:rgb(220,220,220);">
    Figure 6: GIST1M dataset using LSH with the number of hash table vary.
</p>

## lsh原理

LSH(local sensitive hash)

$(d_1,d_2,p_1,p_2)$ - 敏感:

* $d(x,y){\leq}d_1$, 则 $f(x)=f(y)$ 概率至少为 $p_1$.
* $d(x,y){\geq}d_2$, 则 $f(x)=f(y)$ 概率至多为$p_2$.

1. 不同的距离函数对应不同的LSH函数族.
2. 函数族之间有不同的 $(d_1,d_2,p_1,p_2)$ 敏感性.
3. 对一个LSH函数族 $F$, 进行放大处理为 $F'$, 以降低假阳( FP ), 假阴( FN ), 代价是增大了计算时间. $F'$ 的定义为: $F'$ 的每一个成员函数是由 $r$ 个 $F$ 的成员函数构成. 

### 常见的距离与其对应的函数族

* Jaccard 距离对应 $(d_1,d_2,1-d_1,1-d_2)$ LSH 函数族.
* 海明距离对应 $(d_1,d_2,1-d_1/d,1-d_2/d)$LSH 函数族, 除以 $d$ 以归一化为概率.
* 余弦距离对应 $(d_1,d_2,\frac{180-d_1}{180}, \frac{180-d_2}{180})$LSH 函数族, 除以 180 以归一化为概率. 基于随机平面投影. 
* 2维欧式距离对应 $(\frac{a}{2},2a,\frac{1}{2},\frac{1}{3})$LSH 函数族, 需要定义 $a$. 基于随机直线投影, 可以进而推广到多维.

上面参考了[Mining of Massive Datasets cp3](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf).

### FLANN中的LSH浮点数实现

FLANN 中只有海明距离的实现, 先把原数据预处理到海明空间后再建立索引, 因此只能接受整数数据, 想接受浮点数据, 也就是不进行预处理, 应该采用 p-stable LSH. 找到了一个含有 p-stable 的开源实现[$E^2LSH$](http://www.mit.edu/~andoni/LSH/), 包含欧式和海明空间.

上面的内容参考了[LSH和p-stable LSH ](http://blog.sina.com.cn/s/blog_67914f2901019p3v.html).
以及
[Locality-Sensitive Hashing for Finding Nearest Neighbors](http://www.matlabi.ir/wp-content/uploads/bank_papers/g_paper/g15_Matlabi.ir_Locality-Sensitive%20Hashing%20for%20Finding%20Nearest%20Neighbors.pdf).

### p-stable 分布及 p-stable LSH

定义: 随机变量 $X$ 满足对任意的向量 $v$, 有$\sum_iv_iX_i$ 与 $||v||_pX$ 具有相同的分布, 则 $X$ 为 p-stable 分布.

p-stable 分布重要的应用就是可以估计给定向量 $v$ 的 p 范数 $||v||_p$.

在LSH中是反过来用 $(v_1-v_2)a$ 来近似 $||v_1-v_2||_pX$, 此时就可以定义出相应的哈希函数族$h_{a,b}(v)=[\frac{a*v+b}{w}]$, 其中$a*v$表示映射到一条直线上, 然后加$b$除以$w$以分段近似为整数, 依旧是保留了部分信息.

上面的内容参考了 [$E^2LSH$ user manual](http://www.mit.edu/~andoni/LSH/manual.pdf).

### 具体实现

对每一个独立的哈希表, 计算每一个向量$v$对其的键值算法如下.

* 由正态分布$N(0,1)$生成k个向量$a$, 每个向量$a$的维度与输入向量$v$的维度一致, 计算$a*v$, 生成k个哈希值, 对每一个哈希值按照其正负性映射为0,1, 然后按照二进制数连接起来, 获取到哈希表对应的哈希键值.

具体索引相似项时, 因为只是对全部一样的加入候选集会导致候选集过小, 所以计算汉明距离小于等于1的哈希键值对应的项都加入候选集内.


## Reference

[flann](http://www.cs.ubc.ca/research/flann/)

[flann user manual](http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf)

[Mining of Massive Datasets cp3](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf)

[Locality-Sensitive Hashing for Finding Nearest Neighbors](http://www.matlabi.ir/wp-content/uploads/bank_papers/g_paper/g15_Matlabi.ir_Locality-Sensitive%20Hashing%20for%20Finding%20Nearest%20Neighbors.pdf)

[LSH和p-stable LSH ](http://blog.sina.com.cn/s/blog_67914f2901019p3v.html)

[$E^2LSH$](http://www.mit.edu/~andoni/LSH/)

[$E^2LSH$ user manual](http://www.mit.edu/~andoni/LSH/manual.pdf)



