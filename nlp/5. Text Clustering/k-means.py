import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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

# 使用K-means算法进行文本聚类
kmeans = KMeans(n_clusters=2, random_state=0)  # 设置簇的个数为2，可以根据需求调整
kmeans.fit(X)

# 输出每个文档所属的簇标签
labels = kmeans.labels_
for i, label in enumerate(labels):
    print(f"Document {i+1}: Cluster {label}")