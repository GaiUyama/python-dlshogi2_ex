import argparse
import logging
import torch
import torch.optim as optim

from pydlshogi2.network.policy_value_resnet import PolicyValueNetwork
from pydlshogi2.dataloader import HcpeDataLoader

parser = argparse.ArgumentParser(description='Train policy value network')
parser.add_argument('train_data', type=str, nargs='+', help='training data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
parser.add_argument('--testbatchsize', type=int, default=1024, help='Number of positions in each test mini-batch')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--checkpoint', default='checkpoints/checkpoint-{epoch:03}.pth', help='checkpoint file name')
parser.add_argument('--resume', '-r', default='', help='Resume from snapshot')
parser.add_argument('--eval_interval', type=int, default=100, help='evaluation interval')
parser.add_argument('--log', default=None, help='log file path')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('lr={}'.format(args.lr))

# デバイス
if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

dataloader = HcpeDataLoader(args.train_data, args.batchsize, device, shuffle=True, per=True)

dataloader.pre_fetch
