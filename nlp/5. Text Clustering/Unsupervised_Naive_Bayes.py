import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 构建文本数据集
documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document",
    "This is the fourth document"
]

# 使用TF-IDF向量化文本数据
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# 初始化模型参数和文档分配
num_documents = len(documents)
num_features = X.shape[1]
num_clusters = 2
params = {
    'probs': np.ones(num_clusters) / num_clusters,  # 每个模型的先验概率
    'means': np.random.randn(num_clusters, num_features),  # 每个模型的特征均值
    'vars': np.random.randn(num_clusters, num_features),  # 每个模型的特征方差
}
doc_assignments = np.random.randint(num_clusters, size=num_documents)

# 迭代更新模型参数和文档分配
max_iterations = 100
for iteration in range(max_iterations):
    # E步：计算每个文档属于每个模型的后验概率
    posteriors = np.zeros((num_documents, num_clusters))
    for i in range(num_documents):
        doc_features = X[i, :].toarray()
        for j in range(num_clusters):
            prob = params['probs'][j] * np.prod(
                np.exp(-0.5 * ((doc_features - params['means'][j]) ** 2) / (params['vars'][j] + 1e-6))
            )
            posteriors[i, j] = prob
        posteriors[i, :] /= np.sum(posteriors[i, :])

    # M步：更新模型的参数
    for j in range(num_clusters):
        assigned_docs = (doc_assignments == j)
        assigned_samples = X[assigned_docs, :].toarray()
        params['probs'][j] = np.mean(posteriors[:, j])

        if np.sum(assigned_docs) > 0:
            assigned_samples_mean = np.mean(assigned_samples, axis=0)
            assigned_samples_var = np.var(assigned_samples, axis=0)
            params['means'][j] = assigned_samples_mean
            params['vars'][j] = assigned_samples_var

    # 更新文档分配
    doc_assignments = np.argmax(posteriors, axis=1)

# 输出每个文档所属的簇标签
for i, assignment in enumerate(doc_assignments):
    print(f"Document {i+1}: Cluster {assignment}")