import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Example corpus
corpus = [['I', 'love', 'playing', 'soccer'],
          ['Soccer', 'is', 'my', 'favorite', 'sport'],
          ['I', 'enjoy', 'watching', 'soccer', 'matches']]

# Create vocabulary
vocab = {word: i for i, word in enumerate(set([word for sentence in corpus for word in sentence]))}
vocab_size = len(vocab)

# Hyperparameters
embedding_size = 5
window_size = 2
learning_rate = 0.1
epochs = 100

# Generate skip-gram pairs
skip_grams = []
for sentence in corpus:
    if len(sentence) >= 2*window_size+1:  # 确保句子长度足够容纳窗口大小
        for i, word in enumerate(sentence):
            if word in vocab:  # 确保词汇在词汇表中
                target_word = vocab[word]
                context_words = []
                for j in range(i - window_size, i + window_size + 1):
                    if j != i and 0 <= j < len(sentence) and sentence[j] in vocab:  # 确保上下文词汇在词汇表中且不超出句子范围
                        context_words.append(vocab[sentence[j]])
                if context_words:  # 添加此条件来避免跳过没有上下文词汇的目标词汇
                    skip_grams.extend([(target_word, context_word) for context_word in context_words if context_word in vocab])

# ...

# Define SkipGram model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SkipGram, self).__init__()
        self.target_embed = nn.Embedding(vocab_size, embedding_size)
        self.context_embed = nn.Embedding(vocab_size, embedding_size)
        self.activation = nn.LogSoftmax(dim=1)  # 添加激活函数
    
    def forward(self, target, context):
        target_embeds = self.target_embed(target)
        context_embeds = self.context_embed(context)
        target_embeds = self.activation(target_embeds)  # 应用激活函数
        context_embeds = self.activation(context_embeds)  # 应用激活函数
        return target_embeds, context_embeds

# Initialize the model
model = SkipGram(vocab_size, embedding_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()  # 更改为NLLLoss

output_file = open("output.txt", "w")

# Training loop
for epoch in range(epochs):
    np.random.shuffle(skip_grams)
    total_loss = 0
    for target, context in skip_grams:
        target_tensor = torch.tensor(target, dtype=torch.long)
        context_tensor = torch.tensor(context, dtype=torch.long)
        
        optimizer.zero_grad()
        
        target_embeds, context_embeds = model(target_tensor, context_tensor)
        loss = loss_fn(target_embeds, context_tensor)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    output_file.write(f"Epoch {epoch+1}, Loss: {total_loss}\n")

# Get the word embeddings
trained_embeddings = model.target_embed.weight.data.numpy()

# Write the word embeddings to the output file
output_file.write("\nWord Embeddings:\n")
for word, word_id in vocab.items():
    output_file.write(f"{word}: {trained_embeddings[word_id]}\n")

# Close the output file
output_file.close()