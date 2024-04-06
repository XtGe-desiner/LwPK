import argparse
import importlib
from utils import *

MODEL_DIR = None
DATA_DIR = 'D:/CVPR22-Fact-main/data/'
PROJECT = 'fact'


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='mnist',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    # about
    parser.add_argument('-epochs_base', type=int, default=20)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.01)  # cifar 0.1   cub 0.005 imagenet 0.1
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Cosine',
                        choices=['Step', 'Milestone', 'Cosine'])  # cifar  COSINE     CUB  MILESTONE  imagenet COSINE
    parser.add_argument('-milestones', nargs='+', type=int, default=[50, 100, 150, 200, 250, 300])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)  # ViT 0.3 ResNet 0.0005
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)  # CIFAR 0.1  cub 0.25  imagenet 0.1
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot',
                                 'ft_cos'])  # ft_dot means using linear classifier, ft_cos means using cosine
    # classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos',
                                 'avg_cos'])  # ft_dot means using linear classifier, ft_cos means using cosine
    # classifier, avg_cos means using average data embedding and cosine classifier

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-weight_num', type=int, default=20)
    parser.add_argument('-rand_num', type=list, default=[10])
    parser.add_argument('-use_pub', type=bool, default=False)

    # parser.add_argument('-balance', type=float, default=0.001)
    # parser.add_argument('-alpha', type=float, default=0.5)
    # parser.add_argument('-eta', type=float, default=0.1)
    # parser.add_argument('-loss_iter', type=int, default=0)
    # parser.add_argument('-extra_disc', type=str, default="NEW random_num samples")

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
