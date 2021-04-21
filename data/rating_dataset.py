#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 16:40

'制作dataset,把文本数据读入'

__author__ = 'Judgement'

import numpy as np

from torch.utils.data import Dataset
from preprocess import read_csv


class RatingDataSet(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.rating_list = read_csv(data_path)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    @staticmethod
    def collate_fn(batch):
        pass

if __name__ == '__main__':
    dataset = LabeledDataSet()