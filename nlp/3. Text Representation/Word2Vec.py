import gensim.downloader as api

# 下载并加载Google News Word2Vec模型
model = api.load("word2vec-google-news-300")

# 获取单词的词向量
word_vector = model.get_vector('word')

# 计算两个单词的余弦相似度
similarity = model.similarity('apple', 'orange')

# 找到与给定单词最相似的词语
most_similar = model.most_similar('word')

# 寻找与给定词语不同的单词
doesnt_match = model.doesnt_match(['apple', 'orange', 'banana', 'carrot'])

# 寻找与给定词语最相似的词语
similar_by_vector = model.similar_by_vector(word_vector)

# 执行词语间的关系推理
relationship = model.most_similar(positive=['king', 'woman'], negative=['man'])

# 打印结果
print("词向量:", word_vector)
print("余弦相似度:", similarity)
print("最相似的词语:", most_similar)
print("不同的词语:", doesnt_match)
print("最相似的词语（基于词向量）:", similar_by_vector)
print("关系推理结果:", relationship)
