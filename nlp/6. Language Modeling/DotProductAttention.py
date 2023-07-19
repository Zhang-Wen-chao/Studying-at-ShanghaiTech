import torch
import torch.nn as nn

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        weighted_sum = torch.matmul(attention_weights, value.unsqueeze(2)).squeeze(2)
        return weighted_sum, attention_weights


attention = DotProductAttention()
query = torch.tensor([[1.0, 2.0, 3.0]])
key = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
value = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

output, attention_weights = attention(query, key, value)
print("Attention Weighted Sum:")
print(output)
print("Attention Weights:")
print(attention_weights)