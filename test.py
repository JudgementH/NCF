import numpy as np
import torch
from torch.utils.data import DataLoader

from data.binary_dataset import BinaryDataSet
from models.ncf_model import NcfModel
from options.test_option import TestOption
from options.util_option import log

if __name__ == '__main__':
    opt = TestOption().parse()

    dataset = BinaryDataSet(opt.test_filepath,opt.negative_num)
    opt.user_size = dataset.user_size
    opt.movie_size = dataset.movie_size
    log(opt)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    model = NcfModel(opt)

    for i, data in enumerate(dataloader, 0):
        model.set_input(data)
        model.optimize_parameters()
        model.log(i)
