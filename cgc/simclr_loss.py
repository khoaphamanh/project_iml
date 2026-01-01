import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveNTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super(ContrastiveNTXentLoss, self).__init__()

        # parameters
        self.temperature = temperature

    def cosine_similarity(self, z, aug_z):

        z_concat = torch.cat((z, aug_z), dim=0)
        cosine_similarity = torch.matmul(z_concat, z_concat.t()) / self.temperature
        cosine_similarity = cosine_similarity.to(z.device)

        return cosine_similarity

    def forward(self, z: torch.Tensor, z_aug: torch.Tensor):

        # L2 normalize the embeddings
        z = F.normalize(z, dim=1)
        z_aug = F.normalize(z_aug, dim=1)

        # cosine similarity
        cosine_similarity = self.cosine_similarity(z, z_aug)

        # create labels for positive pairs
        N = z.shape[0]
        labels_positive_pairs = torch.arange(N)
        labels_positive_pairs = torch.cat(
            (labels_positive_pairs + N, labels_positive_pairs)
        )
        labels_positive_pairs = labels_positive_pairs.to(z.device)

        # mask the self similarites in diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        cosine_similarity = cosine_similarity.masked_fill(mask, float("-inf"))

        # calculate loss
        loss = F.cross_entropy(cosine_similarity, labels_positive_pairs)

        return loss
