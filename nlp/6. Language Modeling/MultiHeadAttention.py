import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear transformations
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Reshape tensors
        query = query.view(batch_size, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose dimensions to perform matrix multiplication
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.head_dim).float())
        
        # Apply softmax to obtain attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output
        attention_output = torch.matmul(attention_weights, value)
        
        # Transpose dimensions back to original shape
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Linear transformation to get final output
        output = self.output_linear(attention_output)
        
        return output, attention_weights

# Example
attention = MultiHeadAttention(input_dim=256, num_heads=8)
query = torch.randn(32, 50, 256)
key = torch.randn(32, 40, 256)
value = torch.randn(32, 40, 256)

output, attention_weights = attention(query, key, value)
print("Output shape:")
print(output.shape)
print("Attention weights shape:")
print(attention_weights.shape)