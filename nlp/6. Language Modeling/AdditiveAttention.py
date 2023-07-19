import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.query_linear = nn.Linear(query_dim, hidden_dim)
        self.key_linear = nn.Linear(key_dim, hidden_dim)
        self.energy_linear = nn.Linear(hidden_dim, 1)

    def forward(self, query, key, value):
        q = self.query_linear(query)
        k = self.key_linear(key)
        energy = self.energy_linear(torch.tanh(q + k)).squeeze(-1)

        # Use softmax to operate on energy
        attention_weights = torch.softmax(energy, dim=0)
        
        # Reshape attention_weights
        attention_weights = attention_weights.unsqueeze(1)
        
        # Attention weighted sum
        weighted_sum = torch.matmul(value, attention_weights).squeeze(2)
        return weighted_sum, attention_weights.squeeze(1)

# Example
attention = AdditiveAttention(query_dim=3, key_dim=3, hidden_dim=10)
query = torch.tensor([[1.0, 2.0, 3.0]])
key = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
value = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])

output, attention_weights = attention(query, key, value)
print("Attention Weighted Sum:")
print(output)
print("Attention Weights:")
print(attention_weights)