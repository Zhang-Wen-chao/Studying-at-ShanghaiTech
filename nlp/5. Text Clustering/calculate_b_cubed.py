import numpy as np
from collections import defaultdict

def calculate_b_cubed(true_labels, cluster_labels):
    """
    使用B-Cubed评价指标评估文本聚类的一致性

    参数：
    true_labels：真实的标签列表
    cluster_labels：聚类结果的标签列表

    返回：
    B-Cubed评价指标的值
    """

    # 计算准确率和召回率分子
    precision_sum = 0.0
    recall_sum = 0.0

    # 样本数量
    n = len(true_labels)

    # 记录每个簇中每个标签的数量
    clusters = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        clusters[cluster_labels[i]][true_labels[i]] += 1

    # 计算准确率和召回率
    for cluster in clusters.values():
        for label, count in cluster.items():
            precision_sum += (count * (count - 1)) / 2
            recall_sum += (count * (count - 1)) / 2

    # 计算分母
    precision_denom = 0.0
    recall_denom = 0.0
    for cluster in clusters.values():
        cluster_sum = sum(cluster.values())
        precision_denom += (cluster_sum * (cluster_sum - 1)) / 2
        for label, count in cluster.items():
            recall_denom += (count * (count - 1)) / 2

    # 计算准确率和召回率
    precision = precision_sum / precision_denom
    recall = recall_sum / recall_denom

    # 计算B-Cubed评价指标
    b_cubed = (precision + recall) / 2

    return b_cubed


# 真实标签
true_labels = [0, 0, 1, 1, 2, 2]

# 聚类结果
cluster_labels = [1, 1, 0, 0, 2, 0]

# 计算B-Cubed评价指标
b_cubed = calculate_b_cubed(true_labels, cluster_labels)

# 输出评价结果
print("B-Cubed:", b_cubed)