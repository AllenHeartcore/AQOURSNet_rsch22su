import numpy as np
import torch
from dtw import dtw
from kmeans import kmeans
from ts2vec import TS2Vec
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

# series: amount n, length l
# shapelets: amount k, length m

def extract_shapelets(series_set, k, m, enable_ts2vec):
    (_, l) = series_set.shape
    cands = np.concatenate([series_set[:, i : i + m] for i in range(l - m + 1)])
    if enable_ts2vec:
        if cands.ndim == 2: cands = np.expand_dims(cands, axis=2)
        model = TS2Vec(cands.shape[2])
        model.fit(cands)
        cands = model.encode(cands)
    dist, classes = kmeans(torch.from_numpy(cands), k)
    dist = dist.numpy().min(axis=1)
    shapelets = []
    for i in range(k):
        shapelets.append(cands[classes == i][np.argmin(dist[classes == i])])
    return np.stack(shapelets) # shape = (k, m)

def segmented_distance(series, shapelet, enable_dtw):
    (l,), (m,) = series.shape, shapelet.shape
    segments = series[:-(l % m)].reshape(-1, m) # shape = (l / m, m)
    if enable_dtw:
        return np.array([dtw(shapelet, segment).distance for segment in segments])
    else:
        return np.linalg.norm(segments - shapelet, axis=1) # shape = (l / m,)

def embed_series(series_set, shapelets, enable_dtw):
    (n, l), (k, m) = series_set.shape, shapelets.shape
    embedding = np.zeros((n, k, l // m))
    for i, series in tqdm(enumerate(series_set)):
        for j, shapelet in enumerate(shapelets):
            embedding[i, j] = segmented_distance(series, shapelet, enable_dtw)
    return embedding

def adjacency_matrix(embedding, percentile):
    n, k, num_segment = embedding.shape
    embedding = embedding.transpose(0, 2, 1)
    adj_mat = np.zeros((n, k, k))
    for ser_idx in range(n):
        for seg_idx in range(num_segment - 1):
            src_dist = 1. - minmax_scale(embedding[ser_idx, seg_idx])
            dst_dist = 1. - minmax_scale(embedding[ser_idx, seg_idx + 1])
            for src in range(k):
                adj_mat[ser_idx, src] += (src_dist[src] * dst_dist)
    threshold = np.percentile(adj_mat, percentile)
    print("\nthreshold = %f" % threshold)
    return (adj_mat > threshold).astype(np.uint8)
