import argparse
import logging
import os
import time
from types import MethodType

import torch

from .utils.config import update_config

parser = argparse.ArgumentParser(description='HybrIK Training')

"----------------------------- Experiment options -----------------------------"
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    # required=True,
                    default='configs/256x192_adam_lr1e_3_res34_smpl_3d_cam_2x_mix_w_pw3d.yaml',

                    # default='configs/256x192_adam_lr1e_3_hrw48_cam_2x_w_pw3d_3dhp.yaml',
                    type=str)
parser.add_argument('--exp-id', default='test_3dpw', type=str,
                    help='Experiment ID')

"----------------------------- General options -----------------------------"
parser.add_argument('--nThreads', default=1, type=int,
                    help='Number of data loading threads')
parser.add_argument('--snapshot', default=2, type=int,
                    help='How often to take a snapshot of the model (0 = never)')

parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.1.2:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch',
                    help='job launcher')

"----------------------------- Training options -----------------------------"
parser.add_argument('--sync', default=False, dest='sync',
                    help='Use Sync Batchnorm', action='store_true')
parser.add_argument('--seed', default=123123, type=int,
                    help='random seed')
parser.add_argument('--dynamic-lr', default=False, dest='dynamic_lr',
                    help='dynamic lr scheduler', action='store_true')
parser.add_argument('--exp-lr', default=False, dest='exp_lr',
                    help='Exponential lr scheduler', action='store_true')

"----------------------------- Log options -----------------------------"
parser.add_argument('--board', default=True, dest='board',
                    help='Logging with tensorboard', action='store_true')
parser.add_argument('--debug', default=False, dest='debug',
                    help='Visualization debug', action='store_true')
parser.add_argument('--params', default=False, dest='params',
                    help='Logging params', action='store_true')
parser.add_argument('--map', default=True, dest='map',
                    help='Evaluate mAP per epoch', action='store_true')
parser.add_argument('--flip-test',
                    default=True,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--flip-shift',
                    default=False,
                    dest='flip_shift',
                    help='flip shift',
                    action='store_true')


opt = parser.parse_args()
cfg_file_name = opt.cfg.split('/')[-1]
cfg = update_config(opt.cfg)

cfg['FILE_NAME'] = cfg_file_name
cfg.TRAIN.DPG_STEP = [i - cfg.TRAIN.DPG_MILESTONE for i in cfg.TRAIN.DPG_STEP]
num_gpu = torch.cuda.device_count()
if cfg.TRAIN.WORLD_SIZE > num_gpu:
    cfg.TRAIN.WORLD_SIZE = num_gpu
opt.world_size = cfg.TRAIN.WORLD_SIZE
opt.work_dir = './exp/{}/{}-{}/'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id)


if not os.path.exists("./exp/{}/{}-{}".format(cfg.DATASET.DATASET, cfg_file_name, opt.exp_id)):
    os.makedirs("./exp/{}/{}-{}".format(cfg.DATASET.DATASET, cfg_file_name, opt.exp_id))

filehandler = logging.FileHandler(
    './exp/{}/{}-{}/training.log'.format(cfg.DATASET.DATASET, cfg_file_name, opt.exp_id))
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info('\n\n\n\n')
logger.info(time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000)))


def epochInfo(self, set, idx, loss, acc):
    self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
        set=set,
        idx=idx,
        loss=loss,
        acc=acc
    ))

def iterInfo(self, epoch, iter, total_iters, loss_desciption):
    self.info('epoch: {}, iter: {}/{}, loss: {}'.format(epoch, iter, total_iters, loss_desciption))


logger.epochInfo = MethodType(epochInfo, logger)
logger.iterInfo = MethodType(iterInfo, logger)
