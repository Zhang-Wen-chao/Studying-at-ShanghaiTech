import pandas as pd

# 读取CSV文件
df = pd.read_csv('predictions.csv')

# 打印出每列的数据类型
print(df.dtypes)
