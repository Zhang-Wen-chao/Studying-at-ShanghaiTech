from gensim import corpora, matutils
from gensim.models import LdaModel
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

# 转换为Gensim的稀疏矩阵表示
corpus = matutils.Sparse2Corpus(X.T)

# 构建词袋模型
dictionary = corpora.Dictionary.from_corpus(corpus, id2word=dict((i, word) for word, i in vectorizer.vocabulary_.items()))

# 运行LDA模型
num_topics = 2  # 定义主题数
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# 获取文档的主题分布
doc_topics = lda_model.get_document_topics(corpus)

# 输出每个文档所属的簇标签
for i, doc_topic in enumerate(doc_topics):
    cluster = max(doc_topic, key=lambda x: x[1])[0]
    print(f"Document {i+1}: Cluster {cluster}")