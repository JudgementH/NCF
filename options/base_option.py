import argparse
import os


class BaseOption():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--negative_num', type=int, default=4,
                                 help='the number of negative samples for every positive sample')

        self.parser.add_argument('--batch_size', type=int, default=512, help='the size of batch')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

        self.parser.add_argument('--dropout', type=float, default=0.0)
        self.parser.add_argument('--user_embedding_dim', type=int, default=32, help='user embedding layer dim')
        self.parser.add_argument('--movie_embedding_dim', type=int, default=32, help='movie embedding layer dim')
        self.parser.add_argument('--ncf_mf_embedding', type=int, default=32, help='ncf mf layers= embedding dim')
        self.parser.add_argument('--ncf_mlp_embedding', type=int, default=32, help='ncf mlp layers= embedding dim')

        self.parser.add_argument('--ncf_layers', type=int, default=3, help='ncf MLP layer nums')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
