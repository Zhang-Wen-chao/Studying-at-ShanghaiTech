不知道怎么把SMILES数据转换为图数据。
transformer跑了两次，
30 epochs, 0.68104
100 epochs, 0.68491

差不多了，能拿83分了。过几天再说吧。










以下是使用PyTorch Geometric的图神经网络构建示例框架。请注意，这只是一个非常基础的示例，您需要根据自己的数据和任务进行调整。

python
Copy code
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 示例的图神经网络模型
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# 您需要将SMILES数据转换为图数据
# 这里是一个示例，您需要根据您的具体数据进行调整
# nodes_features = ...
# edge_index = ...
# graph_data = Data(x=nodes_features, edge_index=edge_index)

# 创建模型实例
model = GNN(num_node_features=..., num_classes=...)

# 训练和评估模型的步骤类似于之前的过程
在使用GNN之前，您需要确定如何将SMILES字符串转换为图表示。这可能涉及到使用化学信息学的工具，如RDKit，来解析分子结构并提取图形特征。同时，GNN的训练可能涉及到不同的数据处理和优化策略，因此需要在实际应用中进行细致的调整和实验。



