import torch
import torch.nn as nn

# 定义GRU模型
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_seq, targets):
        embedded = self.embedding(input_seq)
        output, hidden = self.gru(embedded)
        output = self.fc(output[:, -1, :])
        
        loss = self.loss_fn(torch.softmax(output, dim=1), targets)
        return output, loss

# 创建GRU模型实例
input_size = 10  # 输入的词汇表大小
hidden_size = 32  # 隐藏层维度
num_layers = 2  # GRU层数
output_size = 2  # 输出类别数

model = GRU(input_size, hidden_size, num_layers, output_size)

# 准备输入数据
input_seq = torch.tensor([[1, 4, 2, 3, 5, 9, 0, 7, 8, 6]])  # 输入序列，形状为(1, sequence_length)
# 1维嵌入向量，每个单词用一个长度为hidden_size的向量表示
model_input = torch.randint(input_size, (1, 10))  # (batch_size, sequence_length)
targets = torch.tensor([0])  # 实际标签

# 设置训练参数
learning_rate = 0.1
num_epochs = 10

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 进行多次迭代训练
for epoch in range(num_epochs):
    # 前向传播和计算损失
    output, loss = model(model_input, targets)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 输出损失
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")