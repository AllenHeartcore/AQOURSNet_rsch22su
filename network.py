import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import xgboost as xgb

class GAT(nn.Module):
    def __init__(self, output_dim, args):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList(
            [GATConv(args.num_segments,            args.hidden_dim, heads=args.heads)] +
            [GATConv(args.hidden_dim * args.heads, args.hidden_dim, heads=args.heads)] * (args.num_layers - 2) +
            [GATConv(args.hidden_dim * args.heads, output_dim, heads=1, concat=False)])
        self.neg_slope = args.neg_slope
        self.dropout = args.dropout
    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.leaky_relu(x, self.neg_slope)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
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
    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        if args.tail == 'none':
            self.gat = GAT(args.num_classes, args)
            self.tail = None
        if args.tail == 'xgboost':
            raise NotImplementedError
        else:
            self.gat = GAT(args.embed_dim, args)
            if args.tail == 'mlp':
                self.tail = MultilayerPerceptron(args.embed_dim, args.num_classes)
            elif args.tail == 'resnet':
                self.tail = FCResidualNetwork(args.embed_dim, args.num_classes)
            else:
                raise NotImplementedError
    def forward(self, x, edge_index, batch):
        x = self.gat(x, edge_index, batch)
        if self.tail is not None: x = self.tail(x)
        return x

def train(model, loader, loss_func, optimizer, args):
    model.train()
    total_loss, num_correct, num_samples = 0., 0, 0
    for batch in loader:
        batch = batch.to(args.device)
        output = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_func(output, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_correct += (output.argmax(dim=1) == batch.y).sum().item()
        num_samples += len(batch.y)
    torch.cuda.empty_cache()
    return total_loss / num_samples, num_correct / num_samples

@torch.no_grad()
def test(model, loader, args):
    model.eval()
    num_correct, num_samples = 0, 0
    for batch in loader:
        batch = batch.to(args.device)
        output = model(batch.x, batch.edge_index, batch.batch)
        num_correct += (output.argmax(dim=1) == batch.y).sum().item()
        num_samples += len(batch.y)
    torch.cuda.empty_cache()
    return num_correct / num_samples

def xgb_train(train_data, train_label, test_data, 
              params={'max_depth': 5, 'eta': 0.1, 'objective': 'binary:logistic'}, num_round=1000):
    train = xgb.DMatrix(train_data, label=train_label)
    test = xgb.DMatrix(test_data)
    booster = xgb.train(params, train, num_round)
    preds = booster.predict(test)
    return preds
