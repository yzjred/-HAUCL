import torch,numpy,random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits, -1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
def simple_batch_graphify(features, lengths):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
    node_features = torch.cat(node_features, dim=0)
    node_features = node_features.cuda()
    return node_features, None, None, None, None
seed = 67137
def seed_everything(seed=seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def _init_fn(worker_id):
    numpy.random.seed(int(seed) + worker_id)
def degree_drop_weights(edge_index, h):
    edge_index_ = edge_index
    deg = degree(edge_index_[1])[:h]
    # deg_col = deg[edge_index[1]].to(torch.float32)
    deg_col = deg
    s_col = torch.log(deg_col)
    # weights = (s_col.max() - s_col+1e-9) / (s_col.max() - s_col.mean()+1e-9)
    weights = (s_col - s_col.min() + 1e-9) / (s_col.mean() - s_col.min() + 1e-9)
    return weights
def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p

    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x
def drop_edge_weighted(edge_index, edge_weights, p: float, h, index, threshold: float = 1.):
    _, edge_num = edge_index.size()
    edge_weights = (edge_weights + 1e-9) / (edge_weights.mean() + 1e-9) * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    # keep probability
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
    edge_remove_index = np.array(list(range(h)))[sel_mask.cpu().numpy()]
    edge_remove_index_all = []
    for remove_index in edge_remove_index:
        edge_remove_index_all.extend(index[remove_index])
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove
    return edge_index
def feature_drop_weights(x, node_c):
    # x = x.to(torch.bool).to(torch.float32)
    x = torch.abs(x).to(torch.float32)
    # 100 x 2012 mat 2012-> 100
    w = x.t() @ node_c
    w = w.log()
    # s = (w.max() - w) / (w.max() - w.mean())
    s = (w - w.min()) / (w.mean() - w.min())
    return s
def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
# batch
def batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / T)
    indices = np.arange(0, num_nodes)
    np.random.shuffle(indices)
    i = 0
    mask = indices[i * batch_size:(i + 1) * batch_size]
    refl_sim = f(sim(z1[mask], z1))  # [B, N]
    between_sim = f(sim(z1[mask], z2))  # [B, N]
    loss = -torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                      / (refl_sim.sum(1) + between_sim.sum(1)
                         - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag()))

    return loss
def com_semi_loss(z1: torch.Tensor, z2: torch.Tensor, T, com_nodes1, com_nodes2):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim[com_nodes1, com_nodes2] / (
                refl_sim.sum(1)[com_nodes1] + between_sim.sum(1)[com_nodes1] - refl_sim.diag()[com_nodes1]))
# loss
def contrastive_loss(x1, x2):
    T = 0.5
    # if args.dname in ["yelp", "coauthor_dblp", "walmart-trips-100"]:
    #     batch_size=1024
    # else:
    #     batch_size = None
    batch_size = None
    if batch_size is None:
        l1 = semi_loss(x1, x2, T)
        l2 = semi_loss(x2, x1, T)
    else:
        l1 = batched_semi_loss(x1, x2, batch_size, T)
        l2 = batched_semi_loss(x2, x1, batch_size, T)
    ret = (l1 + l2) * 0.5
    ret = ret.mean()
    return ret
def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))




class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            best_epoch = []
            for r in result:
                index = np.argmax(r[:, 1])
                best_epoch.append(index)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            print("best epoch:", best_epoch)
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            #             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
def aug_edge(hyperedges_index,removal_probability):
    # [[1, 2, 3], [3, 4, 5, 6], [1, 4, 6, 7]]
    num_edges = hyperedges_index[1].max().item() + 1
    hyperedges = [[] for _ in range(num_edges)]
    for i in range(len(hyperedges_index[0])):
        hyperedges[hyperedges_index[1][i]].append(hyperedges_index[0][i].item())
    #
    # hyperedges = [[1, 2, 3], [3, 4, 5, 6], [1, 4, 6, 7]]
    bipartite_graph, hyperedge_mapping = convert_to_bipartite_hypergraph(hyperedges)
    bipartite_graph1 = random_remove_edges(bipartite_graph, removal_probability)
    hyperedges, _ = convert_to_hypergraph(bipartite_graph1, hyperedge_mapping)

    bipartite_graph1 = random_remove_edges(bipartite_graph, removal_probability)
    hyperedges1, _ = convert_to_hypergraph(bipartite_graph1, hyperedge_mapping)

    # hyperedges = [[1, 2, 3], [3, 4, 5, 6], [1, 4, 6, 7]]
    src = []
    tgt = []
    for i in range(len(hyperedges)):
        for j in range(len(hyperedges[i])):
            src.extend([hyperedges[i][j]])
            tgt.extend([i])
    hyperedges_index = torch.LongTensor([src, tgt])

    src1 = []
    tgt1 = []
    for i in range(len(hyperedges1)):
        for j in range(len(hyperedges1[i])):
            src1.extend([hyperedges1[i][j]])
            tgt1.extend([i])
    hyperedges1_index = torch.LongTensor([src1, tgt1])

    return hyperedges_index,hyperedges1_index
def convert_to_bipartite_hypergraph(hyperedges):
    bipartite_graph = {}
    hyperedge_mapping = {}
    index = 0
    for hyperedge_id, hyperedge in enumerate(hyperedges):
        bipartite_graph[index] = []
        for vertex in hyperedge:
            if vertex not in hyperedge_mapping:
                hyperedge_mapping[vertex] = len(hyperedge_mapping)
            bipartite_graph[index].append(hyperedge_mapping[vertex])
        index += 1
    return bipartite_graph, hyperedge_mapping
def convert_to_hypergraph(bipartite_graph, hyperedge_mapping):
    hyperedges = []
    index_mapping = {}
    for bipartite_id, vertices in bipartite_graph.items():
        hyperedge = []
        for vertex_id in vertices:
            for vertex, mapping_id in hyperedge_mapping.items():
                if mapping_id == vertex_id:
                    hyperedge.append(vertex)
                    break
        hyperedges.append(hyperedge)
        index_mapping[bipartite_id] = len(hyperedges) - 1
    return hyperedges, index_mapping
def random_remove_edges(bipartite_graph, removal_probability):
    edges_to_remove = []
    for hyperedge_id in bipartite_graph:
        if random.random() < removal_probability:
            edges_to_remove.append(hyperedge_id)
    for hyperedge_id in edges_to_remove:
        del bipartite_graph[hyperedge_id]
    return bipartite_graph
def mask_nodes(featrue,aug_ratio):
    node_num = featrue.size()[0]
    feat_dim = featrue.size()[1]
    mask_num = int(node_num * aug_ratio)
    token = featrue.mean(dim=0)
    zero_v = torch.zeros_like(token)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    featrue[idx_mask] = token
    return featrue

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            # a =torch.cat([a , self.pe[:a.size(0)]],dim=-1)
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)