import torch, argparse, warnings
from tqdm import tqdm
from utils import write_log, seed_torch, read_dataset, graph_dataloader
from construct_graph import extract_shapelets, embed_series, adjacency_matrix
from network import NeuralNetwork, train, test

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# DATA LOADING
#parser.add_argument('dataset', help='Path of [.npz] dataset with *ordered* fields "train_data", "train_label", "test_data", "test_label"')
parser.add_argument('--seed',               type=int,   default=42,   help='Random seed')
parser.add_argument('--dataset',            type=str,   default='Coffee', help='Name of UCR dataset')
parser.add_argument('--device',             type=str,   default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

# SHAPELET EMBEDDING
parser.add_argument('--num-shapelets',      type=int,   default=30,   help='Number of shapelets to extract')
parser.add_argument('--num-segments',       type=int,   default=20,   help='Number of segments for shapelet mapping')
parser.add_argument('--pruning-percentile', type=int,   default=30,   help='Percentile for pruning weak edges')

# GAT NETWORK & GRAPH EMBEDDING
parser.add_argument('--hidden-dim',         type=int,   default=256,  help='Hidden dimension of GAT')
parser.add_argument('--embed-dim',          type=int,   default=64,   help='Embedding dimension of graph (output dim. of GAT)')
parser.add_argument('--num-layers',         type=int,   default=4,    help='Number of layers in GAT')
parser.add_argument('--heads',              type=int,   default=8,    help='Number of attention heads in GAT')
parser.add_argument('--neg-slope',          type=float, default=.2,   help='Negative slope of leakyReLU')
parser.add_argument('--dropout',            type=float, default=.5,   help='Dropout rate in training')
parser.add_argument('--tail',               type=str,   default='resnet', help='Type of prediction tail: [none, mlp, resnet]')

# TRAINING & ENHANCEMENT OPTIONS
parser.add_argument('--epochs',             type=int,   default=100,  help='Number of epochs')
parser.add_argument('--batch-size',         type=int,   default=8,    help='Batch size')
parser.add_argument('--lr',                 type=float, default=.001, help='Learning rate')
parser.add_argument('--weight-decay',       type=float, default=.001, help='Weight decay')
parser.add_argument('--ts2vec',             action='store_true', default=False, help='Switch for using TS2VEC')
parser.add_argument('--dtw',                action='store_true', default=False, help='Switch for using DTW')

args = parser.parse_args()
seed_torch(args.seed)

# DATA LOADING & PREPARATION
train_data, train_labels, test_data, test_labels = read_dataset(args.dataset)
setattr(args, 'num_classes', train_labels.max() + 1)
len_shapelet = int(train_data.shape[1] / args.num_segments)

# SHAPELET EMBEDDING & GRAPH GENERATION
shapelets = extract_shapelets(train_data,    len_shapelet,  args)
train_node_features = embed_series(train_data,  shapelets,  args)
test_node_features  = embed_series(test_data,   shapelets,  args)
train_edge_matrices = adjacency_matrix(train_node_features, args)
test_edge_matrices  = adjacency_matrix(test_node_features,  args)
train_loader = graph_dataloader(train_node_features, train_edge_matrices, train_labels, args)
test_loader  = graph_dataloader(test_node_features,  test_edge_matrices,  test_labels,  args)

# GRAPH EMBEDDING & PREDICTION
model = NeuralNetwork(args).to(args.device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
tqdm_meter = tqdm(desc='[Training GAT]')
dataset_name = args.dataset.split('/')[-1].split('.')[0]
log_filename = 'AQOURSNet-log-%s.csv' % dataset_name
write_log(log_filename, 'dataset,epoch,loss,train_acc,test_acc\n')

for epoch in range(args.epochs):
    loss, train_acc = train(model, train_loader, loss_func, optimizer, args)
    test_acc = test(model, test_loader, args)
    tqdm_meter.set_postfix(
        Epoch='%3d' % epoch,
        Loss ='%6f' % loss,
        TrainAcc='%6.2f%%' % (train_acc * 100),
        TestAcc ='%6.2f%%' % (test_acc * 100))
    tqdm_meter.update()
    write_log(log_filename, '%s,%d,%f,%f,%f\n' % (dataset_name, epoch, loss, train_acc, test_acc))
    torch.cuda.empty_cache()
