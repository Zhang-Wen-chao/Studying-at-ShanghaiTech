import pandas as pd

# 读取 CSV 文件
# csv_path = "mini_train.csv"
csv_path = "QSAR_train.csv"

df = pd.read_csv(csv_path)

# 重命名列头
df.rename(columns={"Drug_ID": "name", "Drug": "smiles", "Label": "p_np"}, inplace=True)

# 重新排序列
df = df[['name', 'p_np', 'smiles']]
df.info()
# 保存修改后的 DataFrame 到新的 CSV 文件
df.to_csv("QSAR_train_bbbp.csv", index=False)
