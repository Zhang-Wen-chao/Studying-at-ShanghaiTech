import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

max_length = 1000
embedding_dim = 128
hidden_dim = 64
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class proteindata(Dataset):
    def __init__(self, relations, sequences, max_length=max_length, Train=True):
        self.relations = relations
        self.sequences = sequences
        self.max_length = max_length
        self.has_label = Train
        self.label_encoder = self.create_label_encoder()

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        relation = self.relations.iloc[idx]
        seq_a_encoded = self.get_encoded_sequence(relation['protein_a'])
        seq_b_encoded = self.get_encoded_sequence(relation['protein_b'])

        if self.has_label:
            label = torch.tensor(relation['label'], dtype=torch.float32)
            return (seq_a_encoded, seq_b_encoded, label)
        else:
            return (seq_a_encoded, seq_b_encoded)

    def create_label_encoder(self):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWYU'
        return LabelEncoder().fit(list(amino_acids))

    def get_encoded_sequence(self, protein):
        sequence = self.sequences[protein][:self.max_length]
        encoded = self.label_encoder.transform(list(sequence))
        padded = self.pad_sequence(encoded)
        return torch.tensor(padded, dtype=torch.long)

    def pad_sequence(self, encoded_sequence):
        padding_length = self.max_length - len(encoded_sequence)
        return np.pad(encoded_sequence, (0, padding_length), mode='constant')


def loaddata(file_relation, file_sequences):
    train_relation_df = pd.read_csv(file_relation)
    protein_sequences_df = pd.read_csv(file_sequences)
    sequences_dict = protein_sequences_df.set_index('uniprot_id')['sequence'].to_dict()
    vocab_size = len(set(''.join(protein_sequences_df['sequence']))) + 1
    return train_relation_df, sequences_dict, vocab_size

def preparedata(train_relation_df, sequences_dict):
    dataset = proteindata(train_relation_df, sequences_dict)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class PPI(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(PPI, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(2 * embedding_dim * max_length, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_a, seq_b):
        embed_a = self.embed(seq_a).view(seq_a.size(0), -1)
        embed_b = self.embed(seq_b).view(seq_b.size(0), -1)
        combined = torch.cat((embed_a, embed_b), 1)
        combined = self.dropout(combined)
        out = F.leaky_relu(self.bn1(self.fc1(combined)))
        out = F.leaky_relu(self.bn2(self.fc2(out)))
        out = self.sigmoid(self.fc3(out))
        return out

def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq_a, seq_b, labels in tqdm(train_loader):
            optimizer.zero_grad()
            seq_a, seq_b, labels = seq_a.to(device), seq_b.to(device), labels.to(device)
            outputs = model(seq_a, seq_b).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()  # 更新学习率
        model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for seq_a, seq_b, labels in val_loader:
                seq_a, seq_b, labels = seq_a.to(device), seq_b.to(device), labels.to(device)
                outputs = model(seq_a, seq_b).squeeze()
                predictions = (outputs > 0.5).int() 
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Acc: {accuracy}')
        torch.save(model, 'model.pth')

train_relation = './train_relation.csv'
# train_relation = './small_train.csv'
protein_sequences = './protein_sequences.csv'
train_relation_df, sequences_dict, vocab_size = loaddata(train_relation, protein_sequences)
train_loader, val_loader = preparedata(train_relation_df, sequences_dict)

# 定义学习率调度器参数
step_size = 5  # 每5个epoch减少学习率
gamma = 0.5    # 减少率
epochs = 25
learning_rate = 0.0005

# 初始化优化器和学习率调度器
model = PPI(embedding_dim, hidden_dim, vocab_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train(model, train_loader, val_loader, criterion, optimizer, epochs)

def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for seq_a, seq_b in tqdm(test_loader):
            seq_a, seq_b = seq_a.to(device), seq_b.to(device)
            logits = model(seq_a, seq_b).squeeze()
            probs = torch.sigmoid(logits)
            predictions.extend(probs.cpu().numpy())
    return predictions

def predict_write_csv(predictions, file_path):
    test_df = pd.read_csv(file_path)
    predict_write_csv_df = pd.DataFrame({
        'id': test_df['id'],
        'Predicted_Score': predictions
    })
    predict_write_csv_df.to_csv('predict_output.csv', index=False)

test_df = pd.read_csv('./test_relation.csv')
# test_df = pd.read_csv('./small_test.csv')
test_dataset = proteindata(test_df, sequences_dict, Train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
predictions = predict(model, test_loader)
predict_write_csv(predictions, './test_relation.csv')