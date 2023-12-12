import pandas as pd

# 读取CSV文件
df = pd.read_csv('transformer_100epochs_predictions.csv')

# 删除 'Drug' 和 'Predicted_Label' 列（如果存在）
df.drop(columns=['Drug', 'Predicted_Label'], inplace=True, errors='ignore')

# 将 'Predicted_Probability' 列重命名为 'Predicted_Score'
df.rename(columns={'Predicted_Probability': 'Predicted_Score'}, inplace=True)

# 保存更改后的DataFrame到新文件
df.to_csv('transformer_100epochs_predictions_format.csv', index=False)

# 打印完成消息
print("文件已更新并保存")
