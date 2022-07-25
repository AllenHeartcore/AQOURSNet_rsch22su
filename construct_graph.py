import numpy as np
import torch
from dtw import dtw
from tqdm import tqdm
from kmeans import kmeans
from ts2vec import TS2Vec

# series: amount n, length l
# shapelets: amount k, length m

def extract_shapelets(series_set, len_shapelet, args):
    (_, l) = series_set.shape
    cands = np.concatenate([series_set[:, i : i + len_shapelet] \
                            for i in range(l - len_shapelet + 1)])
    if args.ts2vec:
        if cands.ndim == 2: cands = np.expand_dims(cands, axis=2)
        model = TS2Vec(cands.shape[2], args.device, 
                        hidden_dims=args.ts2vec_dhidden, 
                        output_dims=args.ts2vec_dembed, 
                        depth=args.ts2vec_nlayer)
        model.fit(cands)
        cands = model.encode(cands)
    dist, classes = kmeans(torch.from_numpy(cands), args)
    dist = dist.to('cpu').numpy().min(axis=1)
    classes = classes.to('cpu').numpy()
    if args.device == 'cuda': torch.cuda.empty_cache()
    shapelets = []
    for i in range(args.nshapelet):
        shapelets.append(cands[classes == i][np.argmin(dist[classes == i])])
    return np.stack(shapelets) # shape = (k, m)

def segmented_distance(series, shapelet, args):
    (l,), (m,) = series.shape, shapelet.shape
    if l % m != 0: series = series[:-(l % m)]
    segments = series.reshape(-1, m)[:args.nsegment] # shape = (l / m, m)
    if args.dtw:
        return np.array([dtw(shapelet, segment, 
                            distance_only=True, 
                            dist_method=args.dtw_dist, 
                            step_pattern=args.dtw_step, 
                            window_type=args.dtw_window).distance for segment in segments])
    else:
        return np.linalg.norm(segments - shapelet, axis=1) # shape = (l / m,)

def embed_series(series_set, shapelets, args):
    (n, _), (k, _) = series_set.shape, shapelets.shape
    embedding = np.zeros((n, k, args.nsegment))
    for i, series in tqdm(enumerate(series_set), desc='[Embedding Series]'):
        for j, shapelet in enumerate(shapelets):
            embedding[i, j] = segmented_distance(series, shapelet, args)
    return embedding

def minmax_scale(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def adjacency_matrix(embedding, args):
    print("[Constructing AdjMat]: ", end='')
    n, k, num_segment = embedding.shape
    embedding = embedding.transpose(0, 2, 1)
    adj_mat = np.zeros((n, k, k))
    for ser_idx in range(n):
        for seg_idx in range(num_segment - 1):
            src_dist = 1. - minmax_scale(embedding[ser_idx, seg_idx])
            dst_dist = 1. - minmax_scale(embedding[ser_idx, seg_idx + 1])
            for src in range(k):
                adj_mat[ser_idx, src] += (src_dist[src] * dst_dist)
    threshold = np.percentile(adj_mat, args.percent)
    print("threshold = %f" % threshold)
    return (adj_mat > threshold).astype(np.uint8)
