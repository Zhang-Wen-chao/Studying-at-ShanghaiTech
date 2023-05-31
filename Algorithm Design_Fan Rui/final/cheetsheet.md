# before midterm
## 闫氏dp分析法
分析问题分为状态表示和状态计算两部分
1.状态表示，思考用1维还是2维来储存数据，此题有物品，空间和价值三种数据要储存，所以宜选用2维
           状态表示后又分为集合和属性两部分
        1.1集合：二维数组f[i][j]表示一个怎样的集合? 01背包中表示物品的选法
                考虑前i个物品的情况下，背包容积还剩j时的最优解
        1.2属性：二维数组f[i][j]储存的数据是什么属性 maxn?minn?数目?此题储存的是maxn
（状态表示部分往往是靠经验来写而不是自己想出来的）
2.状态计算：
        将集合划分成子集合，来一层一层获取
        划分的依据是：最后一个不同的节点，如此题是f[i][j]选不选i，选和不选是两种结果
        每层的选与不选可以将集合划分为两个子集，没有遗漏。（求数量时还有遵循不重复的原则）
        选i，[i-1][j-v[i]] + w[i];
        不选[i-1][j];
最后按照分析的结果枚举每一层，输出f[n][m];
# network flow
## max-flow
residual graph、augmenting graph

ford-fulkerson: add a backward path; Worst time complexity: iterations equal to amount of maxflow.

edmonds-karp algorithm ： find the shortest augmenting path. when finding path, regard the residual graph as unweighted. 时间复杂度：边方乘点。
## min-cut
The max-flow min-cut theorem states that the maximum flow of a graph is equal to the capacity of the min-cut in the graph.

run edmonds-karp algorithm to obtain the final residual graph. find the minimum s-t cut（s, t）
## bipartite graph
判定二部图的方法：染色。
### 无权图最大匹配
最大匹配不唯一。
贪心算法可能失败。
转为最大流问题。
### 有权图最大匹配
找最大权重之和。
最大匹配和最小匹配可以相互转化。权重可正可负。
#### hungarian algorithm
hungarian algorithm is for finding the minimum-weight bipartite matching. 
on the graph,the cardinality of u and v must be the same.
time complexity: n^3


# [np-completeness](https://cloud.tencent.com/developer/article/2167585)

p类问题：多项式时间内可解决。
np类问题：多项式时间内可证明。可证明此问题在该输入规模下能在多项式时间内解决。
所有p类问题同时也是np类问题。
np完全问题：如果一个np问题和其他任何np问题一样“不易解决”，那就称为npc类问题，或者np完全问题。
## 证明思路
我们并不是要证明存在某个有效的算法，而是要证明不太可能存在有效的算法。
不知道怎么证明，问题不大，瞎写就行。

基本概念：归约证明
怎么去证明一个问题转换到另一个问题，这是一种规约呢？可按如下几点，分别证明实例的对应性和输出的一致性。
## 几个例子
最短简单路径是简单问题。
最长简单路径是困难的，也只有：确定是否一个图在给定数量的边中包含一条简单路径。这一问题是np完全问题。

我们可以确定是否一个图在O（E）时间内仅仅有一个欧拉回路。
一个有向图中的哈密顿圈是包含V中的每一个顶点的简单回路。
确定一个有向图中是否包含哈密顿圈就是一个np完全问题。

比如，hw4p3：然后我们证明G 是一个有向哈密顿循环当且仅当 C 包含一个简单的 TA 循环。正反说一遍就是证明了。

对于一个布尔公式，若存在对其变量的某种0和1的赋值，使得它的值为1，则此表达式是可满足的。
2-CNF可满足性问题是p类问题。
3-CNF可满足性问题是np完全问题。
3CNF可以规约为团问题
团问题可以规约为顶点覆盖问题
在任意二部图中，最大匹配的边数等于最小顶点覆盖的顶点数。

团问题可以规约为独立集问题
独立集问题可以规约为集合划分问题

子集和问题是np完全的。
Knapsack问题可以拆为子集和问题。
# local search
Local search（局部搜索）是一种在计算机科学和优化领域中常用的启发式算法，用于在大规模搜索空间中寻找最优解或接近最优解的解。它通常用于那些难以直接求解的问题，例如旅行商问题（TSP）、装箱问题（bin packing problem）、集合覆盖问题（set cover problem）等。该算法从一个随机生成的解开始，然后通过反复地尝试对当前解进行“局部”调整来寻找更好的解。具体来说，它会在当前解的邻域内进行搜索，并根据特定的评价准则选择下一个要搜索的解。这个过程会一直持续，直到没有更好的解被找到或者达到了预设的停止条件。

Local search algorithm. Each agent is continually prepared to improve its solution in response to changes made by other agents.

Analogies.
・Nash equilibrium : local search.
・Best response dynamics : local search algorithm.
・Unilateral move by single agent : local neighborhood.

Contrast. Best-response dynamics need not terminate since no single objective function is being optimized.
# Amortized analysis
## Potential method
- Example: Binary counter
## aggregate analysis
## accounting method
算了，放弃。
# Randomized algorithms
Monte Carlo string matching？

## chernoff
Markov’s Inequality
Chebychev’s Inequality

Chernoff bounds 
两个定理抄一下，判断x的性质，根据sigma选公式。

Coin flipping
Load balancing
Set balancing（这不一样吗？【LeetCode】1046. 最后一块石头的重量）
# approximation
和作业题差不多