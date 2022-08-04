import numpy as np
import torch
from tqdm import tqdm

def initialize(X, num_clusters):
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def kmeans(X, num_clusters, args, dtype):
    X = X.float()
    X = X.to(args.device)
    initial_state = initialize(X, num_clusters)
    iteration = 0
    tqdm_meter = tqdm(desc='[Extracting %sShapelets]' % dtype)
    while iteration < args.maxiter:
        dis = pairwise_distance(X, initial_state, args.device)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(args.device)
            selected = torch.index_select(X, 0, selected)
            if args.kmedians: initial_state[index] = selected.median(dim=0).values
            else: initial_state[index] = selected.mean(dim=0)
        center_shift = torch.sum(torch.sqrt(torch.sum( \
            (initial_state - initial_state_pre) ** 2, dim=1)))
        iteration = iteration + 1
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{args.tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < args.tol:
            break
    return dis.cpu(), choice_cluster.cpu(), initial_state.cpu()

def pairwise_distance(data1, data2, device):
    data1, data2 = data1.to(device), data2.to(device)
    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis
