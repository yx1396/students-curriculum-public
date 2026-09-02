import numpy as np


def maf(G):
    """G: (n_samples, n_snps) 剂量 in {0,1,2}. 返回每个 SNP 的最小等位基因频率 (n_snps,)。"""
    G = np.asarray(G, dtype=float)
    p = G.mean(axis=0) / 2.0
    return np.minimum(p, 1.0 - p)


def mac(G):
    """最小等位基因计数 minor allele count (n_snps,)。"""
    G = np.asarray(G, dtype=float)
    ac = G.sum(axis=0)
    n_total = 2 * G.shape[0]
    return np.minimum(ac, n_total - ac)
