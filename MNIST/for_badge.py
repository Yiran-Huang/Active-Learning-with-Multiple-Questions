#The code slightly modified based on the release code https://github.com/JordanAsh/badge/tree/master
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader
from scipy import stats
from functions import *
import numpy as np
def get_embedding(model, X,device, return_probs=False):
    loader_te = DataLoader(CustomDatasetx(X),shuffle=False, batch_size=512)
    model.eval()
    embedding = torch.zeros([X.shape[0], model.get_embedding_dim()])
    probs = torch.zeros(X.shape[0], model.last.out_features)
    count=0
    with torch.no_grad():
        for x, y in loader_te:
            idxs=torch.arange(x.shape[0])+count
            count+=x.shape[0]
            x, y = x.to(device), y.to(device)
            out, e1 = model(x,True)
            embedding[idxs] = e1.data.cpu()
            if return_probs:
                pr = F.softmax(out,1)
                probs[idxs] = pr.data.cpu()
    if return_probs: return embedding, probs
    return embedding
def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = X1_norm_square * X2_norm_square + Y1_norm_square * Y2_norm_square - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist
def init_centers(X1, X2, chosen, chosen_list,  mu, D2,sampling_batch):
    for _ in range(sampling_batch):
        if len(chosen) == 0:
            ind = np.argmax(X1[1] * X2[1])
            mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
            D2 = distance(X1, X2, mu[0]).ravel().astype(float)
            D2[ind] = 0
        else:
            newD = distance(X1, X2, mu[-1]).ravel().astype(float)
            D2 = np.minimum(D2, newD)
            D2[chosen_list] = 0
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in chosen: ind = customDist.rvs(size=1)[0]
            mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
        chosen.add(ind)
        chosen_list.append(ind)
    return chosen, chosen_list, mu, D2
def BadgeSampling(probtest, sampling_batch,device,embtest):
    n_pool=probtest.shape[0]
    idxs_unlabeled = np.arange(n_pool)

    embtest = embtest.numpy()
    probtest = probtest.numpy()

    mu = None
    D2 = None
    chosen = set()
    chosen_list = []
    emb_norms_square = np.sum(embtest ** 2, axis=-1)
    max_inds = np.argmax(probtest, axis=-1)

    probtest = -1 * probtest
    probtest[np.arange(n_pool), max_inds] += 1
    prob_norms_square = np.sum(probtest ** 2, axis=-1)
    #for _ in range(sampling_batch):
    chosen, chosen_list, mu, D2 = init_centers((probtest, prob_norms_square), (embtest, emb_norms_square), chosen, chosen_list, mu, D2,sampling_batch)
    return idxs_unlabeled[chosen_list]