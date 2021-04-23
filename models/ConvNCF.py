import torch
import torch.nn as nn


class ConvNCF(nn.Module):

    def __init__(self, opt):
        super(ConvNCF, self).__init__()
        user_size = opt.user_size
        item_size = opt.item_size

        self.user_embedding = nn.Embedding(user_size, opt.convncf_embedding_dim)
        self.item_embedding = nn.Embedding(item_size, opt.convncf_embedding_dim)

        sequences = []
        pass

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
