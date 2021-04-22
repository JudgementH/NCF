#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 16:49

'对数据基本的预处理工具库'

__author__ = 'Judgement'

import csv

import pandas as pd
import numpy as np


def read_csv(file_path, skip_first=False):
    # read csv into 2d array
    list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        file_reader = csv.reader(f)
        if skip_first:
            _ = next(file_reader)
        for row in file_reader:
            row_list = [int(item) for item in row]
            list.append(row_list)
    return list


def save_csv(array, save_path):
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(array)


def normalize_movlen(movlen_filepath, output_filepath):
    ori_movlen = pd.read_table(movlen_filepath, sep='::', names=["user_id", "movie_id", "rating", "timestamp"],
                               engine='python')
    ori_movlen['user_id'] = ori_movlen['user_id'] - 1
    ori_movlen['movie_id'] = ori_movlen['movie_id'] - 1
    target_df = pd.DataFrame(ori_movlen[['user_id', 'movie_id', 'rating']])
    target_df.to_csv(output_filepath, index=False)


def statistic_csv(csv_filepath):
    df = pd.read_csv(csv_filepath)
    print(df)
    print(df.max())  # user_id max 6040, movie_id max 3952
    print(df.min())  # user_id min 1, movie_id min 1
    list_ = df.values.tolist()
    print(list_[0][0])
    print(list_[0][1])


def pre_test_txt():
    path = '../res/mov_test_with_negative.txt'
    test_data = []
    with open(path, 'r') as f:
        line = f.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1], 1])
            for i in arr[1:]:
                test_data.append([u, int(i), 0])
            line = f.readline()
    print(test_data)
    # save_csv(test_data, '../res/mov_test_normal.csv')


def turn_binary():
    path = '../res/mov_normal_with_negative.csv'
    array = read_csv(path)
    for row in array:
        if row[2] >= 1:
            row[2] = 1

    save_csv(array,'../res/mov_normal_with_negative.csv')

def delete_negative():
    path = '../res/mov_test_normal.csv'
    array = read_csv(path)
    positive = []
    for row in array:
        if row[2] == 1:
            positive.append(row)
    save_csv(positive,'../res/mov_test_positive.csv')

if __name__ == '__main__':
    # normalize_movlen("../res/ratings.dat", "../res/mov_normal.csv")

    # a = np.loadtxt('../res/mov_normal_with_neg.csv', delimiter=',')
    # save_csv(a.astype(np.int32), '../res/mov_normal_with_negative.csv')

    # statistic_csv('../res/mov_normal.csv')

    turn_binary()
