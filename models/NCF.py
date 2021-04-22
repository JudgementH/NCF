#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 16:36

'NCF模型'

__author__ = 'Judgement'

import torch
import torch.nn as nn


class NCF(nn.Module):

    def __init__(self, opt):
        super(NCF, self).__init__()
        user_size = opt.user_size
        movie_size = opt.movie_size

        mf_embedding_dim = opt.ncf_mf_embedding
        self.dropout = opt.dropout

        self.user_mf_embedding = nn.Embedding(user_size, mf_embedding_dim)
        nn.init.normal_(self.user_mf_embedding.weight, std=0.01)
        self.movie_mf_embedding = nn.Embedding(movie_size, mf_embedding_dim)
        nn.init.normal_(self.movie_mf_embedding.weight, std=0.01)

        mlp_embedding_dim = opt.ncf_mlp_embedding
        layers = opt.ncf_layers
        times = 2 ** layers
        self.user_mlp_embedding = nn.Embedding(user_size, mlp_embedding_dim)
        nn.init.normal_(self.user_mlp_embedding.weight, std=0.01)
        self.movie_mlp_embedding = nn.Embedding(movie_size, mlp_embedding_dim)
        nn.init.normal_(self.movie_mlp_embedding.weight, std=0.01)

        sequence = [nn.Linear(mlp_embedding_dim * 2, mlp_embedding_dim * times)]
        for i in range(layers):
            sequence += [nn.Dropout(p=self.dropout),
                         nn.Linear(mlp_embedding_dim * times, mlp_embedding_dim * (times // 2)),
                         nn.ReLU()]
            times = times // 2

        self.mlp = nn.Sequential(*sequence)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        self.neumf_layer = nn.Linear(mf_embedding_dim + mlp_embedding_dim, 1)
        nn.init.kaiming_normal_(self.neumf_layer.weight, a=1, nonlinearity='sigmoid')

        for layer in self.modules():
            if isinstance(layer, nn.Linear) and layer is not None:
                layer.bias.data.zero_()

    def forward(self, user_ids, movie_ids):
        user_mf_embedding = self.user_mf_embedding(user_ids)
        movie_mf_embedding = self.movie_mf_embedding(movie_ids)

        # mf
        mf_output = user_mf_embedding * movie_mf_embedding

        # mlp
        user_mlp_embedding = self.user_mlp_embedding(user_ids)
        movie_mlp_embedding = self.movie_mlp_embedding(movie_ids)
        concat_output = torch.cat((user_mlp_embedding, movie_mlp_embedding), -1)
        mlp_output = self.mlp(concat_output)

        neumf = torch.cat((mf_output, mlp_output), -1)
        output = self.neumf_layer(neumf).view(-1)
        return output
