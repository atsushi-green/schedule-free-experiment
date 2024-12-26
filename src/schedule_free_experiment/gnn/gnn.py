import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Dataset:
    num_features: int
    num_classes: int


class GCN(torch.nn.Module):
    def __init__(self, dataset: Dataset, hidden_channels: int, dropout=0.0):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
