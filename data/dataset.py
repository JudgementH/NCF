#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 16:40

'制作dataset,把文本数据读入'

__author__ = 'Judgement'

from torch.utils.data import Dataset


class LabeledDataSet(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    @staticmethod
    def collate_fn(batch):
        pass