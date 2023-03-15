import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class CL_MLP(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.P = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                      nn.Linear(hidden_dim, hidden_dim),nn.ReLU())





    def forward(self, h1, h2):

        h1 = h1.permute(1, 0, 2)
        h2 = h2.permute(1, 0, 2)

        b, t, h = h1.size()

        h1 = self.P(h1)
        h2 = self.P(h2)

        cl_loss = None

        h_t_1 = torch.cat([h1[:, 1:, :], h2[:, 1:, :]], dim=1)
        h_t_2 = torch.cat([h1[:, :-1, :], h2[:, :-1, :]], dim = 1)

        for i in range(b):

            if cl_loss == None:

                cl_loss = self._InfoNCE(h_t_1[i], h_t_2[i])

            else:

                cl_loss = self._InfoNCE(h_t_1[i], h_t_2[i])


        return h1, h2, cl_loss


    def _InfoNCE(self, emb_i, emb_j, temperature = 0.1):

        b, h = emb_i.size()

        sim = None

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        print('representaetion', representations.size())
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        print('sim mat', similarity_matrix.size())

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]


            numerator = torch.exp(sim_i_j / temperature)
            one_for_not_i = torch.ones((2 * b,)).scatter_(0, torch.tensor([i]), 0.0).to(emb_i.device)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / temperature)
            )

            loss_ij = -torch.log(numerator / denominator)


            return loss_ij.squeeze(0)

        N = b
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss


if __name__ == "__main__":

    CL = CL_MLP(64)
    A = torch.rand((8, 4, 64))
    B = torch.rand((8, 4, 64))

    _, _, L = CL(A,B)

    print(L)
    print(L.size())