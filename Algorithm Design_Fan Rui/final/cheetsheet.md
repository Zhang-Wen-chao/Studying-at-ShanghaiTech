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

# approximation
和作业题差不多

看不懂，要不打印下hw5的solution好了。