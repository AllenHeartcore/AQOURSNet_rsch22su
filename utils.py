import numpy as np
import torch, os, random
from crypt import crypt
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

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

def get_eigenvalue(args, evtype):
    argsval = tuple(args.__dict__.values())
    argsval = [str(i) for i in argsval]
    if evtype == 'shape': argsval = '-'.join(argsval[3:6] + argsval[-9:-5])
    if evtype == 'graph': argsval = '-'.join(argsval[3:9] + argsval[-9:-1])
    if evtype == 'model': argsval = '-'.join(argsval[3:16] + argsval[-9:])
    salt = [str(args.seed)] * 8
    salt = '$1$' + ''.join(salt)
    eigenvalue = crypt(argsval, salt)[12:]
    return eigenvalue.replace('.', '_').replace('/', '+')

def read_dataset(dataset_name):
    try:
        data = np.load('%s.npz' % dataset_name)
    except FileNotFoundError:
        raise FileNotFoundError('Dataset %s not found.' % dataset_name)
    return (data[file] for file in data.files)

class GraphDataset(Dataset):
    def __init__(self, node_features, edge_matrices, labels):
        super(GraphDataset, self).__init__()
        node_features, edge_matrices, labels = map(lambda x: torch.tensor(x), 
                                                   (node_features, edge_matrices, labels))
        self.data = []
        for node_feature, edge_matrix, label in zip(node_features, edge_matrices, labels):
            self.data.append(Data(x=node_feature.float(), 
                                  edge_index=edge_matrix.nonzero().T, 
                                  y=label, num_nodes=node_feature.shape[0]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def graph_dataloader(node_features, edge_matrices, labels, args):
    dataset = GraphDataset(node_features, edge_matrices, labels)
    return DataLoader(dataset, batch_size=args.batchsize, 
                      shuffle=True, num_workers=torch.cuda.device_count())
