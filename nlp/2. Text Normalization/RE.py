import re

# 示例1：匹配文本
text = "Hello, World! How are you?"
pattern = r"Hello"
match = re.search(pattern, text)
if match:
    print("匹配成功")
else:
    print("匹配失败")

# 示例2：替换文本
text = "Hello, World! How are you?"
pattern = r"World"
replacement = "Universe"
new_text = re.sub(pattern, replacement, text)
print(new_text)

# 示例3：提取信息
text = "My email address is john@example.com"
pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
matches = re.findall(pattern, text)
if matches:
    print("找到的邮箱地址：")
    for match in matches:
        print(match)
else:
    print("未找到邮箱地址")
