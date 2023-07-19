import numpy as np
from collections import Counter

def purity_evaluation(true_labels, cluster_labels):
    """
    使用Purity评价指标评估文本聚类的一致性

    参数：
    true_labels：真实的标签列表
    cluster_labels：聚类结果的标签列表

    返回：
    Purity评价指标的值
    """

    # 计算每个聚类中最频繁出现的真实标签
    cluster_purities = []
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        true_labels_in_cluster = np.array(true_labels)[indices]
        label_counts = Counter(true_labels_in_cluster)
        most_common_label = label_counts.most_common(1)[0][1]
        cluster_purities.append(most_common_label / len(indices))

    # 计算加权平均得到总的Purity评价指标
    purity = np.sum(cluster_purities) / len(cluster_labels)

    return purity


# 真实标签
true_labels = [0, 0, 1, 1, 2, 2]

# 聚类结果
cluster_labels = [1, 1, 0, 0, 2, 2]

# 计算Purity评价指标
purity = purity_evaluation(true_labels, cluster_labels)

# 输出评价结果
print("Purity:", purity)