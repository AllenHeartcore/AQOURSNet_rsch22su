import torch, warnings
from tqdm import tqdm
from utils import write_log, seed_torch, ucr_dataset, GraphDataset
from construct_graph import extract_shapelets, embed_series, adjacency_matrix
from network import NeuralNetwork, train, test
from matplotlib import pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATA LOADING & SHAPELET EMBEDDING
dataset_name     = 'Coffee'
log_filename     = 'Time2GraphRework-exp-%s.csv' % dataset_name
num_shapelet     =  30
num_segment      =  20
percentile       =  30

# GAT NETWORK & GRAPH EMBEDDING
hidden_dim       =  32
graph_embed_dim  = 200
num_layers       =   4
heads            =   8
neg_slope        =    .2
dropout          =    .5
tail_type        = 'resnet'

# TRAINING OPTIONS
num_epochs       =  20
learning_rate    =    .0001
weight_decay     =    0

# ENHANCEMENTS
enable_ts2vec    = False ### WARNING: UNSTABLE BUILD
enable_dtw       = False

# DATA LOADING & PREPARATION (utils.py)
seed_torch()
warnings.filterwarnings('ignore')
train_data, train_label, test_data, test_label = ucr_dataset(dataset_name)
if train_label.min() == 1: train_label -= 1
if test_label.min() == 1: test_label -= 1
num_classes = 2
len_shapelet = int(train_data.shape[1] / num_segment)

# SHAPELET EMBEDDING & GRAPH GENERATION (construct_graph.py, ts2vec.py, kmeans.py)
shapelets = extract_shapelets(train_data, num_shapelet, len_shapelet, enable_ts2vec)
for i in range(num_shapelet):
    plt.plot(shapelets[i])
plt.savefig('shapelets.png')
train_node_features = embed_series(train_data, shapelets, enable_dtw)
test_node_features = embed_series(test_data, shapelets, enable_dtw)
train_edge_matrices = adjacency_matrix(train_node_features, percentile)
test_edge_matrices = adjacency_matrix(test_node_features, percentile)
train_set = GraphDataset(train_node_features, train_edge_matrices, train_label)
test_set = GraphDataset(test_node_features, test_edge_matrices, test_label)

# GRAPH EMBEDDING & PREDICTION (network.py)
model = NeuralNetwork(num_segment, hidden_dim, graph_embed_dim, num_classes, 
                      num_layers, heads, neg_slope, dropout, tail_type).to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
tqdm_meter = tqdm(desc='[Training GAT]')
write_log(log_filename, 'epoch,loss,train_acc,test_acc\n')
for epoch in range(num_epochs):
    loss, train_acc = train(model, train_set, train_label, loss_func, optimizer, tail_type)
    test_acc = test(model, test_set, test_label)
    tqdm_meter.set_postfix(
        Epoch='%3d' % epoch,
        Loss='%6f' % loss,
        TrainAcc='%6.2f%%' % (train_acc * 100),
        TestAcc='%6.2f%%' % (test_acc * 100))
    write_log(log_filename, '%d,%f,%f,%f\n' % (epoch, loss, train_acc, test_acc))
    torch.cuda.empty_cache()
