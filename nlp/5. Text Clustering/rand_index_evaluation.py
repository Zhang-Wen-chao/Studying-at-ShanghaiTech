import numpy as np

def rand_index_evaluation(true_labels, cluster_labels):
    """
    使用Rand Index评价指标评估文本聚类的一致性

    参数：
    true_labels：真实的标签列表
    cluster_labels：聚类结果的标签列表

    返回：
    Rand Index评价指标的值
    """

    # 计算样本对在同一簇中或不同簇中的一致性
    total_pairs = len(true_labels) * (len(true_labels) - 1) / 2
    consistent_pairs = 0
    for i in range(len(true_labels)):
        for j in range(i+1, len(true_labels)):
            if true_labels[i] == true_labels[j] and cluster_labels[i] == cluster_labels[j]:
                consistent_pairs += 1
            if true_labels[i] != true_labels[j] and cluster_labels[i] != cluster_labels[j]:
                consistent_pairs += 1

    # 计算Rand Index评价指标
    rand_index = consistent_pairs / total_pairs

    return rand_index


# 真实标签
true_labels = [0, 0, 1, 1, 2, 2]

# 聚类结果
cluster_labels = [1, 1, 0, 0, 2, 0]

# 计算Rand Index评价指标
rand_index = rand_index_evaluation(true_labels, cluster_labels)

# 输出评价结果
print("Rand Index:", rand_index)