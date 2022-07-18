import torch, argparse, warnings
from tqdm import tqdm
from utils import write_log, seed_torch, ucr_dataset, GraphDataset
from construct_graph import extract_shapelets, embed_series, adjacency_matrix
from network import NeuralNetwork, train, test

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# DATA LOADING & SHAPELET EMBEDDING
parser.add_argument('--seed',               type=int,   default=42)
parser.add_argument('--dataset',            type=str,   default='Coffee')
parser.add_argument('--log-filename',       type=str,   default='Time2GraphRework-exp.csv')
parser.add_argument('--num-shapelets',      type=int,   default=30)
parser.add_argument('--num-segments',       type=int,   default=20)
parser.add_argument('--pruning-percentile', type=int,   default=30)

# GAT NETWORK & GRAPH EMBEDDING
parser.add_argument('--hidden-dim',         type=int,   default=256)
parser.add_argument('--embed-dim',          type=int,   default=64)
parser.add_argument('--num-layers',         type=int,   default=4)
parser.add_argument('--heads',              type=int,   default=8)
parser.add_argument('--neg-slope',          type=float, default=0.2)
parser.add_argument('--dropout',            type=float, default=0.5)
parser.add_argument('--tail',               type=str,   default='resnet')

# TRAINING & ENHANCEMENT OPTIONS
parser.add_argument('--epochs',             type=int,   default=100)
parser.add_argument('--lr',                 type=float, default=0.001)
parser.add_argument('--weight-decay',       type=float, default=0.001)
parser.add_argument('--device',             type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--ts2vec',             action='store_true', default=False)
parser.add_argument('--dtw',                action='store_true', default=False)

args = parser.parse_args()
seed_torch(args.seed)

# DATA LOADING & PREPARATION
train_data, train_label, test_data, test_label = ucr_dataset(args.dataset)
setattr(args, 'num_classes', train_label.max() + 1)
len_shapelet = int(train_data.shape[1] / args.num_segments)

# SHAPELET EMBEDDING & GRAPH GENERATION
shapelets = extract_shapelets(train_data,    len_shapelet,  args)
train_node_features = embed_series(train_data,  shapelets,  args)
test_node_features  = embed_series(test_data,   shapelets,  args)
train_edge_matrices = adjacency_matrix(train_node_features, args)
test_edge_matrices  = adjacency_matrix(test_node_features,  args)
train_set = GraphDataset(train_node_features, train_edge_matrices, args)
test_set  = GraphDataset(test_node_features,  test_edge_matrices,  args)

# GRAPH EMBEDDING & PREDICTION
model = NeuralNetwork(args).to(args.device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
tqdm_meter = tqdm(desc='[Training GAT]')
write_log(args.log_filename, 'dataset,epoch,loss,train_acc,test_acc\n')
for epoch in range(args.epochs):
    loss, train_acc = train(model, train_set, train_label, loss_func, optimizer, args)
    test_acc = test(model, test_set, test_label, args)
    tqdm_meter.set_postfix(
        Epoch='%3d' % epoch,
        Loss ='%6f' % loss,
        TrainAcc='%6.2f%%' % (train_acc * 100),
        TestAcc ='%6.2f%%' % (test_acc * 100))
    tqdm_meter.update()
    write_log(args.log_filename, '%s,%d,%f,%f,%f\n' % (args.dataset, epoch, loss, train_acc, test_acc))
    torch.cuda.empty_cache()
