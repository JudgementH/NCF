#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 16:40

'制作dataset,把文本数据读入 rating 0/1'

__author__ = 'Judgement'

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.preprocess import read_csv, save_csv


class BinaryDataSet(Dataset):

    def __init__(self, data_path, negative_num):
        super(BinaryDataSet, self).__init__()
        self.data_path = data_path
        self.negative_num = negative_num
        self.rating_list = np.array(read_csv(data_path))
        self.user_size = max(self.rating_list[:, 0]) + 1
        self.movie_size = max(self.rating_list[:, 1]) + 1

    def make_negative(self, save_path, test_positive_path):
        test_positive = read_csv(test_positive_path)
        self.positive_set = []
        self.negative_set = []
        last_user_id = 0
        for element in self.rating_list:
            user_id = element[0]
            if user_id != last_user_id:
                # make negative
                print("now generating negative sample user_id: %d" % last_user_id)
                for i in range(self.negative_num * len(self.positive_set)):
                    movie_id_negative = np.random.randint(self.movie_size)
                    while movie_id_negative in self.positive_set or movie_id_negative == test_positive[user_id][1]:
                        movie_id_negative = np.random.randint(self.movie_size)
                    self.negative_set.append([last_user_id, movie_id_negative, 0])
                self.positive_set = []
            movie_id_positive = element[1]
            self.positive_set.append(movie_id_positive)
            last_user_id = user_id

        self.negative_set = np.array(self.negative_set)
        self.rating_list = np.concatenate((self.rating_list, self.negative_set), axis=0)
        save_csv(self.rating_list.astype(np.int32), save_path=save_path)

    def __len__(self):
        return len(self.rating_list)

    def __getitem__(self, item):
        return torch.tensor(self.rating_list[item][0]), torch.tensor(self.rating_list[item][1]), torch.tensor(
            self.rating_list[item][2], dtype=torch.float)

    @staticmethod
    def collate_fn(batch):
        user_id = []
        movie_id = []
        rating = []
        for item in batch:
            user_id.append(item[0])
            movie_id.append(item[1])
            rating.append(item[2])
        return torch.tensor(user_id), torch.tensor(movie_id), torch.tensor(rating)


if __name__ == '__main__':
    A = '../res/mov_normal_with_negative.csv'
    B = '../res/dvd_sparse.csv'
    C = '../res/mov_normal.csv'
    dataset = BinaryDataSet(C, 4)
    print(dataset.user_size)
    print(dataset.movie_size)
    print(len(dataset.rating_list))

    dataset.make_negative(save_path='../res/mov_normal_with_negative.csv',
                          test_positive_path='../res/mov_test_positive.csv')

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)
    print(iter(dataloader).next())
