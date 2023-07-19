# 这个错误可能是由于TorchText库版本不匹配造成的。请尝试更新TorchText库版本以解决该问题。可以使用以下命令更新TorchText：
# pip install --upgrade torchtext
# 如果更新后仍然遇到问题，请确保你的PyTorch版本和TorchText版本兼容。你可以查看TorchText的文档以获取更多关于版本兼容性的信息。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# 定义源语言和目标语言的Field
src_language = Field(tokenize='spacy', tokenizer_language='en', lower=True, 
                     init_token='<sos>', eos_token='<eos>')
tgt_language = Field(tokenize='spacy', tokenizer_language='fr', lower=True, 
                     init_token='<sos>', eos_token='<eos>')

# 加载Multi30k数据集
train_data, valid_data, test_data = Multi30k.splits(exts=('.en', '.fr'), 
                                                    fields=(src_language, tgt_language))

# 构建词汇表，并对数据集进行标记化处理
src_language.build_vocab(train_data, min_freq=2)
tgt_language.build_vocab(train_data, min_freq=2)

# 定义模型超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
num_epochs = 10
src_vocab_size = len(src_language.vocab)
tgt_vocab_size = len(tgt_language.vocab)
embedding_size = 256
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.2

# 定义模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size, num_heads, 
                 num_encoder_layers, num_decoder_layers, dropout):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, 
                                          num_decoder_layers, dropout)
        self.fc = nn.Linear(embedding_size, tgt_vocab_size)
        
    def forward(self, src, tgt):
        src_embedding = self.src_embedding(src)
        tgt_embedding = self.tgt_embedding(tgt)
        src_embedding = src_embedding.permute(1, 0, 2)
        tgt_embedding = tgt_embedding.permute(1, 0, 2)
        outputs = self.transformer(src_embedding, tgt_embedding)
        outputs = outputs.permute(1, 0, 2)
        predictions = self.fc(outputs)
        return predictions

# 初始化模型和优化器
model = Transformer(src_vocab_size, tgt_vocab_size, embedding_size, num_heads, 
                    num_encoder_layers, num_decoder_layers, dropout).to(device)
optimizer = optim.Adam(model.parameters())

# 定义损失函数
criterion = nn.CrossEntropyLoss(ignore_index=tgt_language.vocab.stoi['<pad>'])

# 定义训练和评估函数
def train(model, iterator, optimizer, criterion):
    model.train()
    loss_epoch = 0
    for batch in iterator:
        src = batch.src.to(device)
        tgt = batch.trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, tgt[:-1, :])
        
        output = output.view(-1, tgt_vocab_size)
        tgt = tgt[1:].view(-1)
        
        loss = criterion(output, tgt)
        
        loss.backward()
        
        optimizer.step()
        
        loss_epoch += loss.item()
        
    return loss_epoch / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    loss_epoch = 0
    with torch.no_grad():
        for batch in iterator:
            src = batch.src.to(device)
            tgt = batch.trg.to(device)

            output = model(src, tgt[:-1, :])

            output = output.view(-1, tgt_vocab_size)
            tgt = tgt[1:].view(-1)

            loss = criterion(output, tgt)
            
            loss_epoch += loss.item()
        
    return loss_epoch / len(iterator)

# 创建数据迭代器
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=batch_size, 
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)

# 开始训练模型
for epoch in range(num_epochs):
    train_loss = train(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

# 在测试集上评估模型
test_loss = evaluate(model, test_iterator, criterion)
print(f"Test Loss: {test_loss:.4f}")