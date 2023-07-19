import nltk
from nltk import ngrams
from collections import defaultdict

# 训练数据
data = "I am learning about N-gram models. N-gram models are widely used in natural language processing. They help in predicting the probability of the next word given a sequence of words. N-gram models can be used for various applications such as speech recognition and machine translation. They are based on the assumption of Markov property. Markov property assumes that the probability of a word depends only on the previous n-1 words. The value of n determines the order of the model. Higher order models take into account more context but require more training data. N-grams can suffer from sparsity issues when encountering unseen combinations of words. Smoothing techniques are often used to address this problem."

tokens = nltk.word_tokenize(data)

# 构建N-gram模型
n = 2  # 二元模型
ngram_model = ngrams(tokens, n)

# 统计频数
ngram_freq = defaultdict(int)
for grams in ngram_model:
    context = grams[:-1]  # 前文
    ngram_freq[grams] += 1
    ngram_freq[context] += 1
ngram_freq = dict(ngram_freq)

# 估计概率
ngram_prob = {}
for grams, freq in ngram_freq.items():
    context = grams[:-1]  # 前文
    grams_str = ' '.join(grams)  # 处理为字符串
    context_str = ' '.join(context)  # 处理为字符串
    denominator = ngram_freq.get(context, 0) + len(ngram_freq)
    if context:
        ngram_prob[grams_str] = freq / denominator
        ngram_prob[context_str] = ngram_freq[context] / denominator
    else:
        ngram_prob[grams_str] = freq / denominator

# 输入前文，预测下一个词
context = ("I",)
next_word = None
max_prob = 0
for grams, prob in ngram_prob.items():
    if tuple(grams.split())[:-1] == context and prob > max_prob:
        max_prob = prob
        next_word = tuple(grams.split())[-1]

print("Given context: ", context)
print("Next word prediction: ", next_word)