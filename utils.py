import numpy as np
import torch, os, random
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def write_log(log_filename, message):
    with open(log_filename, 'a') as log:
        log.write(message)

def seed_torch(seed=42):
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
    def __init__(self, node_features, edge_matrices, labels):
        super(GraphDataset, self).__init__()
        node_features, edge_matrices, labels = map(lambda x: torch.tensor(x).to(device), 
                                                   (node_features, edge_matrices, labels))
        self.data = []
        for node_feature, edge_matrix, label in zip(node_features, edge_matrices, labels):
            self.data.append(Data(x=node_feature.float(), 
                                  edge_index=matrix2index(edge_matrix), 
                                  y=label, num_nodes=node_feature.shape[0]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class GraphDataLoader(DataLoader):
    def __init__(self, node_features, edge_matrices, labels, batch_size):
        super(GraphDataLoader, self).__init__(GraphDataset(node_features, edge_matrices, labels), 
                                              batch_size=batch_size, shuffle=True)
