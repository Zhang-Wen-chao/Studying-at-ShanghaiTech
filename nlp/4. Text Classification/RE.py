import re

# 构建训练数据集
training_data = [
    ("Please send me an email.", "email"),
    ("Call me on my phone number.", "phone"),
    # 其他示例...
]

# 创建正则表达式模式
patterns = {
    "email": r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b",
    "phone": r"\b(\+\d{1,2}\s?)?((\(\d{3}\))|\d{3})(\s|-)?\d{3}(\s|-)?\d{4}\b",
    # 其他模式...
}

# 文本分类函数
def classify_text(text):
    for category, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            return category
    return "unknown"

# 测试
test_text = "Please send me an email."
category = classify_text(test_text)
print(f"The category of the text '{test_text}' is: {category}")