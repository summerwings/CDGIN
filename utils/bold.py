import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from random import randrange


def process_dynamic_fc(timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True, seed = 0, stat_des = 'Mean'):
    # assumes input shape [minibatch x time x node]
    # output shape [minibatch x time x node x node]
    if dynamic_length is None:
        dynamic_length = timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(timeseries.shape[1]-dynamic_length+1)
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

    dynamic_fc_list = []
    for i in sampling_points:
        fc_list = []
        for _t in timeseries:
            fc = corrcoef(_t[i:i+window_size].T, stat_des, seed)
            if not self_loop: fc -= torch.eye(fc.shape[0])
            fc_list.append(fc)
        dynamic_fc_list.append(torch.stack(fc_list))
    return torch.stack(dynamic_fc_list, dim=1), sampling_points


# corrcoef based on
# https://github.com/pytorch/pytorch/issues/1254
def corrcoef(x, stat_des, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if stat_des == 'Cor':
        stat_x = torch.mean(x, 1, keepdim=True)
        xs = x.sub(stat_x.expand_as(x))
        c = xs.mm(xs.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        c = torch.clamp(c, -1.0, 1.0)

    if stat_des == 'Sim':
        c = x.cpu().detach().numpy()
        c = - euclidean_distances(c)
        c = torch.from_numpy(c).float().to(device)

    return c
