import pandas as pd

# Step 1: Read the CSV file
file_path = 'QSAR_test.csv'  # 替换为您的CSV文件路径
df = pd.read_csv(file_path)

# 重命名列头
df.rename(columns={"Drug_ID": "name", "Drug": "smiles"}, inplace=True)

# Step 2: Add a new column 'p_np' with all values set to 0
df['p_np'] = 0

df = df[['name', 'p_np', 'smiles']]
df.info()

# Step 3: Save the updated DataFrame to a new CSV file
updated_file_path = 'QSAR_test_bbbp.csv'  # 您可以指定其他保存路径和文件名
df.to_csv(updated_file_path, index=False)
