from options.base_option import BaseOption


class TrainOption(BaseOption):

    def __init__(self):
        BaseOption.__init__(self)

        self.parser.add_argument('--train_filepath', type=str, default='./res/train.csv',
                                 help='the path of test file')
        self.parser.add_argument('--test_filepath', type=str, default='./res/test.csv', help='test file path')
        self.parser.add_argument('--checkpoint_root', type=str, default='./checkpoint/')

        self.parser.add_argument('--epoch', type=int, default=20, help='epoch')
        self.parser.add_argument('--top_k', type=int, default=10, help='evaluate top k')
        self.parser.add_argument('--test_batch_size', type=int, default=100)
