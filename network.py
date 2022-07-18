import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import xgboost as xgb

class GAT(nn.Module):
    def __init__(self, output_dim, kwargs):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList(
            [GATConv(kwargs.num_segments,              kwargs.hidden_dim, heads=kwargs.heads)] +
            [GATConv(kwargs.hidden_dim * kwargs.heads, kwargs.hidden_dim, heads=kwargs.heads)] * (kwargs.num_layers - 2) +
            [GATConv(kwargs.hidden_dim * kwargs.heads, output_dim, heads=1, concat=False)])
        self.neg_slope = kwargs.neg_slope
        self.dropout = kwargs.dropout
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.leaky_relu(x, self.neg_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=512, dropouts=[.1, .2, .2, .3], num_layers=4):
        super(MultilayerPerceptron, self).__init__()
        if isinstance(hidden_sizes, int): hidden_sizes = [hidden_sizes] * (num_layers - 1)
        if isinstance(dropouts, float): dropouts = [dropouts] * num_layers
        assert len(hidden_sizes) == num_layers - 1
        assert len(dropouts) == num_layers
        in_channels = [input_size] + hidden_sizes
        out_channels = hidden_sizes + [output_size]
        tails = [nn.ReLU()] * (num_layers - 1) + [nn.Softmax(dim=0)]
        layers = []
        for dropout, in_channel, out_channel, tail in zip(dropouts, in_channels, out_channels, tails):
            layers.extend([nn.Dropout(dropout), nn.Linear(in_channel, out_channel), tail])
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class FCResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_sizes=64, num_layers=3):
        super(FCResidualBlock, self).__init__()
        if isinstance(hidden_sizes, int): hidden_sizes = [hidden_sizes] * (num_layers - 1)
        assert len(hidden_sizes) == num_layers - 1
        in_channels = [input_size] + hidden_sizes
        out_channels = hidden_sizes + [input_size]
        layers = []
        for in_channel, out_channel in zip(in_channels, out_channels):
            layers.extend([nn.Linear(in_channel, out_channel), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x) + x

class FCResidualNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 128, 128], num_blocks=3, num_layers=3):
        super(FCResidualNetwork, self).__init__()
        if isinstance(hidden_sizes, int): hidden_sizes = [hidden_sizes] * num_blocks
        if isinstance(num_layers, int): num_layers = [num_layers] * num_blocks
        assert len(hidden_sizes) == num_blocks
        assert len(num_layers) == num_blocks
        blocks = []
        for (hidden_size, num_layer) in zip(hidden_sizes, num_layers):
            blocks.append(FCResidualBlock(input_size, hidden_size, num_layer))
        blocks.extend([nn.Linear(input_size, output_size), nn.Softmax(dim=0)])
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)

class NeuralNetwork(nn.Module):
    def __init__(self, kwargs):
        super(NeuralNetwork, self).__init__()
        if kwargs.tail == 'none' or 'xgboost':
            self.gat = GAT(kwargs.num_classes, kwargs)
            self.tail = None
        else:
            self.gat = GAT(kwargs.embed_dim, kwargs)
            if kwargs.tail == 'mlp':
                self.tail = MultilayerPerceptron(kwargs.embed_dim, kwargs.num_classes)
            elif kwargs.tail == 'resnet':
                self.tail = FCResidualNetwork(kwargs.embed_dim, kwargs.num_classes)
            else:
                raise NotImplementedError
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index).mean(0)
        if self.tail is not None: x = self.tail(x)
        return x

def train(model, dataset, labels, loss_func, optimizer, kwargs):
    model.train()
    outputs = []
    labels = torch.tensor(labels).to(kwargs.device)
    for graph in dataset:
        outputs.append(model(graph.x, graph.edge_index))
    outputs = torch.stack(outputs)
    loss = loss_func(outputs, labels.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if kwargs.tail == 'xgboost':
        raise NotImplementedError
    return loss.item(), (outputs.argmax(1) == labels).float().mean().item()

@torch.no_grad()
def test(model, dataset, labels, kwargs):
    model.eval()
    outputs = []
    labels = torch.tensor(labels).to(kwargs.device)
    for graph in dataset:
        outputs.append(model(graph.x, graph.edge_index).argmax())
    return (torch.stack(outputs) == labels).float().mean().item()

def xgb_train(train_data, train_label, test_data, 
              params={'max_depth': 5, 'eta': 0.1, 'objective': 'binary:logistic'}, num_round=1000):
    train = xgb.DMatrix(train_data, label=train_label)
    test = xgb.DMatrix(test_data)
    booster = xgb.train(params, train, num_round)
    preds = booster.predict(test)
    return preds
