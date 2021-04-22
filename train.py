import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.binary_dataset import BinaryDataSet
from models.ncf_model import NcfModel
from options.train_option import TrainOption
from options.util_option import log

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':
    opt = TrainOption().parse()

    dataset = BinaryDataSet(opt.train_filepath, opt.negative_num)
    test_dataset = BinaryDataSet(opt.test_filepath, opt.negative_num)
    opt.user_size = dataset.user_size
    opt.movie_size = dataset.movie_size
    log(opt)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=dataset.collate_fn,
                            num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                 collate_fn=dataset.collate_fn)
    model = NcfModel(opt)
    print(model.model)

    HR_max = 0
    NDCG_max = 0
    times = 0
    for epoch in range(opt.epoch):
        dataloader = tqdm(dataloader)
        dataloader.set_description(
            '[%s%04d/%04d %s=%f]' % ('Epoch:', epoch + 1, opt.epoch, 'lr', model.opt.learning_rate))

        # train
        model.model.train()
        loss_sum = 0
        for i, data in enumerate(dataloader, 0):
            model.set_input(data)
            model.optimize_parameters()
            loss_sum += model.loss.item()
            dataloader.set_postfix({'loss': loss_sum / (i + 1)})

        model.model.eval()
        HR, NDCG = model.test(test_dataloader)

        if HR > HR_max:
            # save the model
            HR_max = HR
            NDCG_max = NDCG
            times = epoch
            model.save_model(opt.checkpoint_root, f'NCF_epoch{epoch}')

    print(f'save the best HR model in epoch {times} with HR:{HR_max} and NDCG:{NDCG_max}')
