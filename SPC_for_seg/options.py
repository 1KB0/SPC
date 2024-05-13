import argparse
from networks.models import DenseNet121


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='D:/Bai/Datasets/PEAD/', help='dataset root dir') # for PEAD

    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')

    parser.add_argument('--base_lr', type=float,  default=0.0001, help='maximum epoch number to train')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--local_ep', type=int,  default=1, help='local epoch')
    parser.add_argument('--num_users', type=int,  default=3, help='num_users') #for_PEAD
    parser.add_argument('--rounds', type=int,  default=800, help='rounds')

    ### tune
    parser.add_argument('--resume', type=str,  default=None, help='model to resume')
    parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
    parser.add_argument('--global_step', type=int,  default=0, help='global_step')
    ### costs
    parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--consistency', type=float,  default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')
    args = parser.parse_args()
    return args
