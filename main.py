import torch, argparse, warnings
from utils import seed_torch, read_dataset, graph_dataloader
from construct_graph import extract_shapelets, embed_series, adjacency_matrix
from network import NeuralNetwork
from process import process

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# DATA LOADING
parser.add_argument('dataset',     type=str,   help='Name of UCR dataset')
parser.add_argument('--seed',      type=int,   default=42,       help='Random seed')
parser.add_argument('--device',    type=str,   default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

# SHAPELET EMBEDDING
parser.add_argument('--nshapelet', type=int,   default=30,       help='Number of shapelets to extract')
parser.add_argument('--nsegment',  type=int,   default=20,       help='Number of segments for shapelet mapping')
parser.add_argument('--maxiter',   type=int,   default=300,      help='Max number of iterations of KMeans')
parser.add_argument('--tol',       type=float, default=1e-4,     help='Tolerance of KMeans')
parser.add_argument('--percent',   type=int,   default=30,       help='Percentile for pruning weak edges')

# GAT NETWORK
parser.add_argument('--dhidden',   type=int,   default=256,      help='Hidden dimension of GAT')
parser.add_argument('--dembed',    type=int,   default=64,       help='Embedding dimension of graph (output dim. of GAT)')
parser.add_argument('--nlayer',    type=int,   default=4,        help='Number of layers in GAT')
parser.add_argument('--nhead',     type=int,   default=8,        help='Number of attention heads in GAT')
parser.add_argument('--negslope',  type=float, default=.2,       help='Negative slope of leakyReLU')
parser.add_argument('--dropout',   type=float, default=.5,       help='Dropout rate')
parser.add_argument('--tail',      type=str,   default='linear', help='Type of prediction tail: [none, linear, mlp, resnet]')

# TRAINING
parser.add_argument('--nepoch',    type=int,   default=100,      help='Number of epochs')
parser.add_argument('--nbatch',    type=int,   default=16,       help='Number of mini-batches')
parser.add_argument('--optim',     type=str,   default='Adam',   help='Optimization algorithm for learning')
parser.add_argument('--lr',        type=float, default=.001,     help='Learning rate')
parser.add_argument('--wd',        type=float, default=.001,     help='Weight decay')

# ENHANCEMENT OPTIONS
parser.add_argument('--ts2vec',           action='store_true',   default=False,        help='Switch for using TS2VEC')
parser.add_argument('--ts2vec-dhidden',   type=int,              default=64,           help='Hidden dimension of TS2Vec encoder')
parser.add_argument('--ts2vec-dembed',    type=int,              default=320,          help='Embedding dimension of TS2Vec encoder')
parser.add_argument('--ts2vec-nlayer',    type=int,              default=10,           help='Number of layers in TS2Vec encoder')
parser.add_argument('--dtw',              action='store_true',   default=False,        help='Switch for using DTW')
parser.add_argument('--dtw-dist',         type=str,              default='euclidean',  help='Pointwise distance function of DTW')
parser.add_argument('--dtw-step',         type=str,              default='symmetric2', help='Local warping step pattern of DTW')
parser.add_argument('--dtw-window',       type=str,              default='none',       help='Windowing function of DTW')

args = parser.parse_args()
seed_torch(args.seed)

# DATA LOADING
train_data, train_labels, test_data, test_labels = read_dataset(args.dataset)
setattr(args, 'nclass', train_labels.max() + 1)
setattr(args, 'batchsize', len(train_data) // args.nbatch)
setattr(args, 'nameset', args.dataset.split('/')[-1].split('.')[0])
setattr(args, 'namelog', 'output/AQOURSNet-log-%s.txt' % args.nameset)
len_shapelet = int(train_data.shape[1] / args.nsegment)

# SHAPELET EMBEDDING & GRAPH GENERATION
shapelets = extract_shapelets(train_data,    len_shapelet,  args)
train_node_features = embed_series(train_data,  shapelets,  args)
test_node_features  = embed_series(test_data,   shapelets,  args)
train_edge_matrices = adjacency_matrix(train_node_features, args)
test_edge_matrices  = adjacency_matrix(test_node_features,  args)

# GAT NETWORK & TRAINING
model = NeuralNetwork(args).to(args.device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = eval('torch.optim.%s' % args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wd)
train_loader = graph_dataloader(train_node_features, train_edge_matrices, train_labels, args)
test_loader = graph_dataloader(test_node_features, test_edge_matrices, test_labels, args)
process(model, train_loader, test_loader, loss_func, optimizer, args)
