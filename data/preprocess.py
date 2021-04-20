#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 16:49

'对数据基本的预处理工具库'

__author__ = 'Judgement'

import pandas as pd


def normalize_movlen(movlen_filepath, output_filepath):
    ori_movlen = pd.read_table(movlen_filepath, sep='::', names=["user_id", "movie_id", "rating", "timestamp"],
                               engine='python')
    print(ori_movlen[['user_id','movie_id','rating']])


if __name__ == '__main__':
    normalize_movlen("../res/ratings.dat", "../res/mov_normal.csv")
