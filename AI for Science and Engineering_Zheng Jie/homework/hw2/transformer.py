import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 载入数据
# train_data = pd.read_csv('./mini_train.csv')
# test_data = pd.read_csv('./mini_test.csv')
train_data = pd.read_csv('./QSAR_train.csv')
test_data = pd.read_csv('./QSAR_test.csv')

# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 数据预处理
class QSARDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, include_labels=True):
        self.tokenizer = tokenizer
        self.texts = dataframe['Drug']
        self.include_labels = include_labels
        if self.include_labels:
            self.labels = dataframe['Label']
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_length,
          return_token_type_ids=False,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        item = {
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.include_labels:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 设置最大序列长度
MAX_SEQ_LEN = 128

# 使用类时，对于训练数据集，包含标签
train_dataset = QSARDataset(train_data, tokenizer, MAX_SEQ_LEN, include_labels=True)

# 对于测试数据集，不包含标签
test_dataset = QSARDataset(test_data, tokenizer, MAX_SEQ_LEN, include_labels=False)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Transformer模型
class QSARTransformer(nn.Module):
    def __init__(self):
        super(QSARTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # 增加额外的输出层以进行分类
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2) # 假设是二分类

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 取[CLS]标记的输出用于分类任务
        cls_output = outputs.pooler_output
        logits = self.classifier(cls_output)
        return logits

# 检查CUDA是否可用
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 将模型移至设定的设备（GPU或CPU）
model = QSARTransformer().to(device)
# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # 将数据移至相同的设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        # 运行模型
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


# 模型评估 - 获取预测概率
model.eval()
pred_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # 获取正类的概率
        pred_probs.append(probs.cpu())

# 将预测概率列表转换为Tensor
pred_probs = torch.cat(pred_probs)

# 转换概率为预测类别，可选阈值为0.5
pred_labels = (pred_probs >= 0.5).int()

# 保存预测概率和/或预测类别到文件
# 例如：保存为CSV文件
test_data['Predicted_Probability'] = pred_probs.numpy()
test_data['Predicted_Label'] =pred_labels.numpy() 
test_data.to_csv('predictions100.csv', index=False)