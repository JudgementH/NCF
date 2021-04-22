import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.NCF import NCF
from utils import evaluate


class NcfModel():

    def __init__(self, opt):
        self.opt = opt

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = NCF(opt).to(self.device)
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.criterion = nn.BCEWithLogitsLoss()

    def set_input(self, data):
        user_ids, movie_ids, ratings = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
        self.user_ids = user_ids.cuda()
        self.movie_ids = movie_ids.cuda()
        self.ratings = ratings.cuda()

    def forward(self):
        self.pred_ratings = self.model(self.user_ids, self.movie_ids)

    def backward(self):
        self.loss = self.criterion(self.pred_ratings, self.ratings)
        self.loss.backward()

    def optimize_parameters(self):
        self.model.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()
        # self.scheduler.step()

    def test(self, test_loader):
        HR, NDCG = [], []

        test_loader = tqdm(test_loader)
        test_loader.set_description('[test]')
        for data in test_loader:
            self.set_input(data)
            self.forward()
            topk = min(self.pred_ratings.shape[0], self.opt.top_k)
            _, indices = torch.topk(self.pred_ratings, topk)
            recommends = torch.take(self.movie_ids, indices).cpu().numpy().tolist()
            gt_item = self.movie_ids[0].item()

            hr = evaluate.hit(gt_item, recommends)
            ndcg = evaluate.ndcg(gt_item, recommends)
            HR.append(hr)
            NDCG.append(ndcg)
            test_loader.set_postfix({'HR': np.mean(HR), 'NDCG': np.mean(NDCG)})
        return np.mean(HR), np.mean(NDCG)

    def log(self, i):
        print(i, self.loss.item())

    def save_model(self, save_path, name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filepath = os.path.join(save_path, f'{name}.pth')
        torch.save(self.model, filepath)
