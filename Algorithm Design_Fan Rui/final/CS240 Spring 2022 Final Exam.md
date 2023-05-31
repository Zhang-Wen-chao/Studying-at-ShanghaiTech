<a name="br1"></a>ShanghaiTech University Final Examination Cover Sheet

Academic Year : 2022 Term: Spring

Course-offering School: SIST

Instructor: Rui Fan

Course Name: Algorithm Design and Analysis /

Course Number: CS 240

General Instructions:

1\. All examination rules must be strictly observed throughout the entire test. Any form of

academic dishonesty is strictly prohibited, and violations may result in severe sanctions.

2\. Based on government policy, you will be videorecorded during the entire examination, and

the recordings will be provided to the government for possible inspection.

3\. This exam is closed book. You may only use 2 pages of notes. You may not use your

computer or phone for any purposes other than downloading the exam, submitting your

solutions and for videorecording.

Exam Instructions:

1\. In all problems in which you are asked to design algorithms, you should clearly describe

how your algorithm works, and provide pseudocode if this improves clarity. You need to

argue why your algorithm is correct, and analyze your algorithm when asked to.

2\. All answers must be written neatly and legibly in English.

# 1 Answer true or false for the following.
(10 points, 2 points each)
## 1a
Let $f$ be a maximum $s-t$ flow on a graph $G=(V,E)$, and let $B$ be the set of vertices in $V$ which can reach $t$ in the residual graph $G_f$. Then $(V-B,B)$ is a minimum capacity $s-t$ cut.
### maxflow
True.

Let $A=V-B$. Since $B$ is the set of vertices in $V$ which can reach $t$ in the residual graph $G_f$, we know that $s\rightarrow V\setminus B\rightarrow B\rightarrow t$ is a path in $G_f$. This means that there is no edge from $A$ to $B$ in the residual graph $G_f$, otherwise we can augment the flow by sending flow along this edge, contradicting the maximality of $f$. Therefore, any $s-t$ path in $G$ must contain at least one edge from $A$ to $B$.

Now consider any cut $(A,B')$ of $G$. Since $B'$ is the set of vertices that are unreachable from $s$ in the subgraph induced by $A$ and $B'$, we have $s\rightarrow A\rightarrow B'$. This means that there is an $s-t$ path $(s\rightarrow A\rightarrow B'\rightarrow t)$ in $G$ that contains edges from both $A$ to $B'$ and $B$ to $V\setminus B'$, so the capacity of the cut is at least the flow passing through these edges. But since there is no edge from $A$ to $B$ in $G_f$, all the flow passing through edges from $A$ to $B$ must already be saturated, so the capacity of the cut is at least the capacity of the edges from $B$ to $V\setminus B'$ in the residual graph $G_f$. Therefore, the minimum cut has capacity at least $\text{cap}(B,V\setminus B')$, i.e., $(V-B,B)$. 

Since $(V-B,B)$ is an $s-t$ cut and its capacity is equal to the maximum flow, it is a minimum capacity $s-t$ cut.


## 1b
Let $f$ be a maximum $s-t$ flow on a graph $G=(V,E)$, and let $(S,T)$ be the corresponding minimum capacity $s-t$ cut. Then every edge $e$ which crosses between $S$ and $T$ has flow $f(e)$ equal to its capacity c(e).
### maxflow
True.

Let $(S,T)$ be the minimum capacity $s-t$ cut corresponding to the maximum flow $f$. Suppose for the sake of contradiction that there exists an edge $e=(u,v)$ with $u\in S$ and $v\in T$ such that $f(e) < c(e)$, i.e., the flow on $e$ is less than its capacity. Since $u\in S$ and $v\in T$, it follows that all $s-t$ paths in $G$ must contain an edge from $S$ to $T$.

Consider the residual graph $G_f$, where $f$ denotes the maximum flow on $G$. Since $f(e) < c(e)$, we have $c_f(e)=c(e)-f(e)>0$. By the Ford-Fulkerson/Edmonds-Karp algorithm, there exists an $s-t$ path $P$ in $G_f$ with positive capacity. Since $P$ is a path in $G_f$, we know that $P$ does not contain any edges $(u',v')$ with $u'\in S$ and $v'\in T$, because these edges would saturate under the maximum flow $f$. Therefore, $P$ consists entirely of edges either in $S$ or in $T$, but not both.

Now consider the cut $(S',T')$ induced by $P$, where $S'=\{u: u \in S \text{ and }$ there exists a path in $P$ from $u$ to $s\}$ and $T'=V\setminus S'$. Since all edges on $P$ are in either $S'$ or $T'$, it follows that the capacity of the cut $(S',T')$ is equal to the capacity of $P$, which is positive.

On the other hand, since $S'$ is a subset of $S$, and all $s-t$ paths contain an edge from $S$ to $T$, it follows that there can be no $s-t$ path in $G$ that goes through $(S',T')$. Therefore, the capacity of $(S',T')$ is less than the capacity of the minimum cut $(S,T)$, which contradicts the assumption that $(S,T)$ is the minimum capacity $s-t$ cut. Thus, we must have $f(e)=c(e)$ for all edges that cross between $S$ and $T$.

## 1c

If a problem $B$ is NP-hard, and problem $A \le _p B$, then $A$ is NP-hard.
### np
Yes, that statement is correct. 

To see why, let's first define what it means for a problem to be NP-hard. A problem is NP-hard if every problem in the complexity class NP can be reduced to it in polynomial time. In other words, if there is an efficient way to solve the NP-hard problem, then there is an efficient way to solve any problem in NP.

Now, suppose we have a problem $B$ that is NP-hard, and a problem $A$ such that $A \le_q B$, where $\le_q$ denotes polynomial-time reducibility. This means that there exists a polynomial-time algorithm that can transform instances of problem $A$ into instances of problem $B$, such that the solution to the transformed instance of $B$ can be used to find the solution to the original instance of $A$.

Since $B$ is NP-hard and $A \le_q B$, we can reduce any instance of $A$ to an instance of $B$ in polynomial time. Therefore, if we had an efficient algorithm to solve $A$, we could use the reduction from $A$ to $B$ followed by an efficient algorithm for solving $B$ to solve any NP problem in polynomial time. This implies that $A$ is also NP-hard.


## 1d
If a problem $A\in P$, then $A \le _p B$ for every problem $B\in NP$.

### np
This statement is not necessarily true. 

Recall that a problem $A$ is in the complexity class P if there exists an algorithm that can solve instances of $A$ in polynomial time. On the other hand, a problem $B$ is in the complexity class NP if there exists a non-deterministic polynomial-time algorithm that can recognize instances of $B$, or equivalently, if there exists a polynomial-time algorithm that can verify solutions to instances of $B$.

If $A \in P$, it means that we have an efficient algorithm that can solve instances of $A$. However, this does not necessarily mean that there exists a polynomial-time reduction from $A$ to every problem in NP. In fact, there are many problems in NP that are believed to be harder than problems in P, such as the famous example of the Traveling Salesman Problem (TSP).

Therefore, it is possible for $A \in P$ but for there to exist a problem $B \in NP$ such that $A \nleq_p B$. This would mean that we do not know how to reduce instances of $A$ to instances of $B$ efficiently, even though we can solve instances of $A$ efficiently.
## 1e

Let $T(n)$ be the complexity of a problem on input size $n$, and suppose $T(n)=T(\frac{9n}{10}) + T(\frac{3n}{20}) + O(n)$. Then $T(n)=O(n)$.
### 递归树
This statement is not necessarily true. The given recurrence relation is known as a divide-and-conquer recurrence, which typically applies to problems that can be divided into subproblems of smaller sizes and then combined. 

To solve this recurrence relation using the master theorem, we need to determine the values of $a$, $b$, and $f(n)$ in the equation $T(n) = a T(\frac{n}{b}) + f(n)$. In this case, we have:

- $a=2$ because the recurrence has two recursive calls.
- $b=\frac{10}{9}$ because the larger subproblem has size $\frac{9n}{10}$.
- $f(n)=O(n)$ because the work done outside the recursive calls is $O(n)$.

Now, we can use the master theorem to determine the asymptotic complexity of $T(n)$. Specifically, we need to compare $f(n)$ to $n^{\log_b a}$. We have:

$$\log_b a = \log_{\frac{10}{9}} 2 \approx 3.17$$

Since $f(n)=O(n)$, we can apply case 1 of the master theorem, which gives us a complexity of $T(n) = \Theta(n^{\log_b a}) = \Theta(n^{3.17})$. Therefore, we cannot conclude that $T(n)=O(n)$.



# 2
(10 points)

Given a sequence of numbers, you want to find a subsequence whose values are all increasing, and whose sum is maximum. For example, given the sequence $[8,4,15,7,14,12]$, the optimal subsequence is $[4,7,14]$, which has sum 25. Designan efficient algorithm for this problem, and analyze its complexity.

## dynamic programming
This problem can be solved using dynamic programming. Let $S_i$ denote the maximum sum of an increasing subsequence ending at index $i$. Then we have the recurrence relation:

$$S_i = \max_{j<i, a_j < a_i} \{S_j + a_i\}$$

which means we consider all indices $j$ that are less than $i$ and have a smaller value than $a_i$, and pick the one that maximizes the sum $S_j+a_i$. 

To find the overall maximum sum increasing subsequence, we take the maximum over all possible ending indices:

$$\max_{i=1}^n \{S_i\}$$

We can initialize $S_1=a_1$ for the base case, and then compute the values of $S_i$ for $i=2,3,\ldots,n$ in order using the recurrence relation above.

Here's the pseudocode for the algorithm:

```
input: a sequence of n numbers a_1, a_2, ..., a_n

# initialize base case
S[1] = a[1]

# compute max sum increasing subsequence ending at each index i
for i from 2 to n:
    max_sum = 0
    for j from 1 to i-1:
        if a[j] < a[i]:
            max_sum = max(max_sum, S[j] + a[i])
    S[i] = max_sum

# find the overall maximum sum increasing subsequence
max_sum = 0
for i from 1 to n:
    max_sum = max(max_sum, S[i])

output: the maximum sum of any increasing subsequence in the input sequence
```

The time complexity of this algorithm is $O(n^2)$, since we need to compute $S_i$ using a nested loop over $j$. However, it is possible to optimize this algorithm using binary search, which reduces the time complexity to $O(n \log n)$. The idea is to maintain a list of increasing subsequence endings and use binary search to find the largest one that can be extended by the current number. This approach will require sorting the sequence and maintaining additional arrays, so it is more complicated than the basic dynamic programming approach.


# 3
You and a group of friends have borrowed money from each other, and now it’s time

for everyone to pay what they owe. Your goal is to do this while mimimizing the

total amount of money transferred. For example, in the left figure below, each edge

$(u,v)$ indicates that $u$ owes $v$ the number shown on the edge. If each person pays

all the amounts indicated in the left figure, the total amount of money transferred is 9

RMB. However, the right figure shows an equivalent way for everyone to get their

money, which only transfers a total of 7 RMB, and this amount is optimal. Design

an efficient algorithm to minimize the total amount of money transferred, argue why

your algorithm is correct, and analyze the complexity of your algorithm.

(10 points)

## maxflow
This problem can be modeled as a minimum-cost flow problem, where each person is a node, and each debt between two people is an edge with a weight equal to the amount owed. We can assume each person starts with a balance of 0.

To solve this problem, we can use the Edmonds-Karp algorithm, which is a variant of the Ford-Fulkerson algorithm, to find the maximum flow in a network with a minimum cost. In this case, the maximum flow we want to find is the total amount owed, and the minimum cost is the minimum amount of money transferred.

First, we need to construct a directed graph with a source node s and a sink node t. For each person, we add a node to the graph with an edge from s to that person's node with a capacity equal to the total amount that person owes. We also add an edge from each person's node to t with a capacity equal to the total amount that person is owed.

For each debt between two people, we add an edge from the node corresponding to the person who owes money to the node corresponding to the person who is owed money with a capacity equal to the amount owed and a cost equal to the negative of the amount owed. This represents the transfer of money from the person who owes to the person who is owed. 

Now, we can run the Edmonds-Karp algorithm on this graph to find the maximum flow from s to t with a minimum cost. The flow network gives us a solution to the problem: each edge $(u,v)$ with a positive flow corresponds to a transfer of money from $u$ to $v$ equal to the amount of flow on the edge.

The total amount transferred is equal to the total cost of the flow, which is negative because we set the costs of the edges to be the negative of the amount owed. Therefore, we need to multiply the total cost by -1 to get the actual amount transferred.

The time complexity of the Edmonds-Karp algorithm is O(VE^2), where V and E are the number of nodes and edges in the graph, respectively. In this case, V is equal to the number of people, and E is equal to the number of debts. Therefore, the time complexity of our algorithm is O(N^2M^2), where N is the number of people and M is the number of debts. However, there are more efficient algorithms for solving the minimum-cost flow problem, such as the network simplex algorithm and the successive shortest path algorithm, which can reduce the time complexity to O(NMlogN) and O(NM^2logN), respectively.

# 4
Suppose you are given an array of ꢖ numbers and want to find the ꢫ’th largest

number, for some ꢤ ꢒ ꢫ ꢒ ꢖ. There are deterministic algorithms to solve this

problem in ꢠꢆꢖꢊ time, but they are somewhat complicated. Design a simple

randomized algorithm to solve this problem in ꢠꢆꢖꢊ expected time. Clearly

describe the algorithm and analyze its expected time complexity.

Hint: Use an approach similar to randomized Quicksort.

(10 points)

solution4:The algorithm that you are looking for is called randomized selection. It is very similar to randomized quicksort, as you mentioned, and it can find the th smallest element in an unsorted array of n elements in expected O(n) time.

The basic idea of the algorithm is to select a pivot element at random, partition the array around the pivot, and then recursively apply the algorithm to one of the partitions until the desired element is found.

Here's how the algorithm works:

1.Choose a pivot element from the array uniformly at random.
2.Partition the array into two subarrays: one containing all elements smaller than the pivot, and another containing all elements greater than or equal to the pivot.
3.If the index of the pivot element in the sorted array is k (i.e., we are looking for the kth largest element), then we either return the pivot element if k equals its index, or apply the algorithm recursively to the appropriate subarray otherwise.
The expected running time of this algorithm is O(n), because at each step we reduce the size of the problem by roughly half (on average), just like in randomized quicksort. The worst-case running time is still O(n^2), but this occurs with very low probability.

Note that the constant factors in the O(n) expected running time may be larger than those in deterministic algorithms, but randomized selection is much simpler to implement and has better asymptotic behavior.

# 5
Recall from class that we found a 2-approximation algorithm for the ꢫ-center problem.

You will now prove this is essentially optimal, by showing that there does not exist a polynomial time ꢆꢧ ꢂ ꢬꢊ-approximation algorithm for ꢫ -center, for any ꢬ ꢭ ꢮ , unless P = NP.

To do this, we first consider ꢫ-center as a graph problem. Let ꢄ be a complete,
 undirected, weighted graph on vertex set ꢇ, where the weights satisfy the triangle
 inequality. That is, for any ꢩꢈ ꢪꢈ ꢯ Ȃ ꢇ, there exist edges ꢆꢩꢈ ꢪꢊꢈ ꢆꢩꢈ ꢯꢊꢈ ꢆꢪꢈ ꢯꢊ in ꢄ,
 and ꢰꢆꢩꢈ ꢪꢊ ꢒ ꢰꢆꢩꢈ ꢯꢊ ꢝ ꢰꢆꢯꢈ ꢪꢊ, where ꢰꢆꢏꢊ denotes the weight of an edge ꢏ. For any node ꢪ Ȃ ꢇ and set ꢍ Ȃ ꢇ, define ꢰꢆꢪꢈ ꢍꢊ ꢅ ꢱꢲꢳ ꢰꢆꢩꢈ ꢪꢊꢶ The ꢫ-center

ꢴȂꢵ

problem is to find a set of ꢫ nodes ꢍ Ȃ ꢇ such that ꢱꢷꢸ ꢰꢆꢪꢈ ꢍꢊ is the minimum,

ꢹȂꢺ

among all possible choices for ꢍ. Let ꢰȂ ꢅ ꢱꢷꢸ ꢰꢆꢪꢈ ꢍꢊ be the value for this optimal

ꢹȂꢺ

set ꢍ.

` `Next, we define the ꢫ-dominating set problem. Here, we are given an undirected,
unweighted, and not necessarily complete graph ꢄ ꢅ ꢆꢇꢈ ꢉꢊ, and need to determine
 whether there exists a set of ꢫ nodes ꢍ Ȃ ꢇ such that Ȃꢪ Ȃ ꢇȂꢩ Ȃ ꢍꢻ ꢆꢩꢈ ꢪꢊ Ȃ ꢉ.
 That is, we want to know if there are ꢫ nodes such that every node in ꢇ is adjacent
 to one of the ꢫ nodes. For example, the red nodes in the graph below are a 2- dominating set, and this graph does not have a 1-dominating set.

It is known that the ꢫ-dominating set problem is NP-complete (when ꢫ is part of the

input). We use this to prove ꢫ-center is NP-hard to approximate to factor ꢧ ꢂ ꢬ. In

particular, suppose for contradiction there exists a polytime ꢆꢧ ꢂ ꢬꢊ-approximation

algorithm ꢑ for ꢫ-center, for some ꢬ ꢭ ꢮ. That is, for any instance of ꢫ-center, if

the optimal solution value is ꢰȂ, ꢑ returns a value ꢰꢼ ꢒ ꢆꢧ ꢂ ꢬꢊꢰȂ. Then, given

an instance ꢄ of ꢫ-dominating set, describe how to construct in polytime an instance

ꢁ




<a name="br4"></a>ꢄ′ of ꢫ-center based on ꢄ, so that by running ꢑ on ꢄ′, you can determine whether

ꢄ has a solution in polytime. Note that you need to define the vertex set of ꢄ′ and

the weights of all its edges, and also describe how the output of ꢑ tells you whether

ꢄ is solvable.

Hint: It suffices to make all the weights in ꢄ′ be either 1 or 2.

(10 points)

END OF EXAM

4
