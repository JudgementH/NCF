from options.base_option import BaseOption


class TestOption(BaseOption):

    def __init__(self):
        BaseOption.__init__(self)

        self.parser.add_argument('--test_filepath', type=str, default='./res/mov_normal_with_negative.csv',help='the path of test file')
