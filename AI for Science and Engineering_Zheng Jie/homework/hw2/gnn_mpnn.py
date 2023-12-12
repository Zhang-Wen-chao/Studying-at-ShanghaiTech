import torch
import torch.nn as nn
import dgl
from dgllife.utils import smiles_to_bigraph
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from dgl.nn.pytorch import MPNNPredictor

# 确保DG-LifeSci库已安装
# pip install dgllife dgl rdkit

# 将SMILES字符串转换为图的函数
def smiles_to_graph(smiles_string):
    if isinstance(smiles_string, str):
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None
        # 直接从SMILES字符串生成图
        return smiles_to_bigraph(smiles_string)
    else:
        return None

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels

# 创建数据集类
class QSARDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe['graph'] = self.dataframe['Drug'].apply(smiles_to_graph)
        self.dataframe = self.dataframe[self.dataframe['graph'].notnull()]
        self.graphs = self.dataframe['graph'].tolist()
        self.labels = torch.tensor(self.dataframe['Label'].values, dtype=torch.long)  # 将标签转换为长整型
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

class QSARMPNN(nn.Module):
    def __init__(self, node_input_dim=74, edge_input_dim=12, output_dim=1):
        super(QSARMPNN, self).__init__()
        self.mpnn = MPNNPredictor(node_in_feats=node_input_dim,
                                  edge_in_feats=edge_input_dim,
                                  node_out_feats=64,
                                  edge_hidden_feats=64,
                                  num_step_message_passing=3,
                                  num_step_set2set=3,
                                  num_layer_set2set=3)

        self.fc = nn.Linear(64, output_dim)

    def forward(self, g, n_feats, e_feats):
        node_feats, graph_feats = self.mpnn(g, n_feats, e_feats)
        return torch.sigmoid(self.fc(graph_feats)).squeeze(1)

# 加载数据
# train_data = pd.read_csv('QSAR_train.csv')
# test_data = pd.read_csv('QSAR_test.csv')
train_data = pd.read_csv('mini_train.csv')
test_data = pd.read_csv('mini_test.csv')
# 创建数据集和数据加载器
train_dataset = QSARDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)


# 模型、损失函数和优化器
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = QSARGNN(in_feats=1, hidden_size=64, num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batched_graph, labels in train_loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        logits = model(batched_graph)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 模型训练后进行训练集评估
model.eval()
train_pred_probs = []
train_labels_list = []
with torch.no_grad():
    for batched_graph, labels in train_loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)

        logits = model(batched_graph)
        probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]  # 获取正类的概率
        train_pred_probs.append(probs.cpu())
        train_labels_list.append(labels.cpu())

# 将预测概率列表和标签列表转换为Tensor
train_pred_probs = torch.cat(train_pred_probs)
train_labels = torch.cat(train_labels_list)

# 计算ROC AUC值
from sklearn.metrics import roc_auc_score
train_roc_auc = roc_auc_score(train_labels.numpy(), train_pred_probs.numpy())
print(f"Training ROC AUC: {train_roc_auc}")

# 测试集预测
test_data['graph'] = test_data['Drug'].apply(smiles_to_graph)
test_graphs = test_data['graph'].tolist()
test_dataset = dgl.batch(test_graphs).to(device)

model.eval()
with torch.no_grad():
    logits = model(test_dataset)
    predicted_probs = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()

# 保存预测结果
predictions = pd.DataFrame({
    'Drug_ID': test_data['Drug_ID'],
    'Predicted_Score': predicted_probs
})
predictions.to_csv('mpnn_predicted_results.csv', index=False)
