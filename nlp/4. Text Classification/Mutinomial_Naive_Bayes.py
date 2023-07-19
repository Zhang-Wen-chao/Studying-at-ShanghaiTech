from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 构建训练数据集
training_data = [
    ("Please send me an email.", "email"),
    ("Call me on my phone number.", "phone"),
    ("I need help with my laptop.", "computer"),
    ("What's the weather today?", "weather"),
]
# 分离文本和标签
texts, labels = zip(*training_data)

# 实例化文本向量化器
vectorizer = CountVectorizer()

# 向量化训练文本
X = vectorizer.fit_transform(texts)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 实例化多项式朴素贝叶斯分类器
classifier = MultinomialNB()

# 训练分类器
classifier.fit(X_train, y_train)

# 对测试集进行预测
predicted = classifier.predict(X_test)

# 输出分类报告
print(classification_report(y_test, predicted))