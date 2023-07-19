from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 构建训练数据集
training_data = [
    ("Please send me an email.", "email"),
    ("Call me on my phone number.", "phone"),
    ("I need help with my laptop.", "computer"),
    ("What's the weather today?", "weather"),
]
# 分离文本和标签
texts, labels = zip(*training_data)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义文本向量化函数
def vectorize_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)[0]
    embeddings = torch.mean(outputs, dim=1).cpu().numpy()
    return embeddings

# 向量化训练文本
X_train_vec = vectorize_text(X_train)

# 重塑训练数据的形状
X_train_vec = np.array(X_train_vec).reshape(-1, 1)

# 训练分类器
classifier = LogisticRegression()
classifier.fit(X_train_vec, y_train)

# 向量化测试文本并进行预测
X_test_vec = np.array(vectorize_text(X_test)).reshape(-1, 1)
predicted = classifier.predict(X_test_vec)

# 输出分类报告
print(classification_report(y_test, predicted))