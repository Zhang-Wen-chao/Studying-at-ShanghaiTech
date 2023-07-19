import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze()
        x = self.fc(x)
        return x

# 创建Transformer模型实例
input_dim = 512
hidden_dim = 2048
output_dim = 10
num_heads = 8
num_layers = 6

model = Transformer(input_dim, hidden_dim, output_dim, num_heads, num_layers)

# 将模型移至GPU设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 创建输入数据
batch_size = 16
sequence_length = 100
input_data = torch.randn(batch_size, sequence_length, input_dim).to(device)

# 前向传播
output = model(input_data)
print(output.shape)