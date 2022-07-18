import numpy as np
import torch, os, random
from torch_geometric.data import Data, Dataset
from dgl import graph

def write_log(log_filename, message):
    with open(log_filename, 'a') as log:
        log.write(message)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ucr_dataset(dataset_name):
    data = np.load('/data/chenzy/ucr/%s.npz' % dataset_name)
    return (data[file] for file in data.files)

def matrix2index(matrix):
    return matrix.nonzero().T

class GraphDataset(Dataset):
    def __init__(self, node_features, edge_matrices, kwargs):
        super(GraphDataset, self).__init__()
        node_features, edge_matrices = map(lambda x: torch.tensor(x).to(kwargs.device), 
                                           (node_features, edge_matrices))
        self.data = []
        for node_feature, edge_matrix in zip(node_features, edge_matrices):
            self.data.append(Data(x=node_feature.float(), 
                                  edge_index=matrix2index(edge_matrix), 
                                  num_nodes=node_feature.shape[0]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
