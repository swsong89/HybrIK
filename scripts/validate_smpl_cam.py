"""Validation script."""
import argparse
import os
import pickle as pk
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from hybrik.datasets import HP3D, PW3D, H36mSMPL
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.env import init_dist
from hybrik.utils.metrics import NullWriter
from hybrik.utils.transforms import get_func_heatmap_to_coord
from tqdm import tqdm

parser = argparse.ArgumentParser(description='HybrIK Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default='configs/256x192_adam_lr1e_3_hrw48_cam_2x_w_pw3d_3dhp_r9000p.yaml',
                    # required=True,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    default='pretrained_model/hybrik_hrnet48_w3dpw.pth',
                    # required=True,
                    type=str)
parser.add_argument('--gpu',
                    help='gpus',
                    default = '0',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    default='16',
                    type=int)
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
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed testing')
parser.add_argument('--dist-url', default='tcp://127.0.1.2:23456', type=str,
                    help='url used to set up distributed testing')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch',
                    help='job launcher')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed testing')

opt = parser.parse_args()
cfg = update_config(opt.cfg)
# 设置gpu
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

# 添加根目录
currentfile = os.path.abspath(__file__)  # '/home/ssw/code/ik/Hybrik/opt.py'
if 'data2' in currentfile: # 3090
    dataset_dir = '/data2/2020/ssw/dataset'
elif 'code' in currentfile: # r9000p
    dataset_dir = '/home/ssw/code/dataset'
    opt.print_freq = 1
else:
    raise IOError('dataset {} not exists.'.format(currentfile))

cfg['DATASET']['DATASET_DIR'] = dataset_dir

norm_method = cfg.LOSS.get('norm', 'softmax')


def validate_gt(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=32, pred_root=False, test_vertice=False):
    print('opt.flip_test: ', opt.flip_test)
    print('batch_size: ', batch_size)

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=False, sampler=gt_val_sampler)
    kpt_pred = {}
    m.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])
    pve_list = []

    for inps, labels, img_ids, img_paths, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].cuda(opt.gpu)
            except AttributeError:
                assert k == 'type'

        # output = m(inps, trans_inv, intrinsic_param, joint_root, depth_factor, (gt_betas, None, None))
        output = m(inps, flip_test=opt.flip_test, bboxes=bboxes,
                   img_center=labels['img_center'])
        if test_vertice:
            gt_betas = labels['target_beta']
            gt_thetas = labels['target_theta']
            gt_output = m.module.forward_gt_theta(gt_thetas, gt_betas)

        pred_uvd_jts = output.pred_uvd_jts
        pred_xyz_jts_24 = output.pred_xyz_jts_24.reshape(inps.shape[0], -1, 3)[:, :24, :]
        pred_xyz_jts_24_struct = output.pred_xyz_jts_24_struct.reshape(inps.shape[0], 24, 3)
        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)
        pred_mesh = output.pred_vertices.reshape(inps.shape[0], -1, 3)
        if test_vertice:
            gt_mesh = gt_output.vertices.reshape(inps.shape[0], -1, 3)
            gt_xyz_jts_17 = gt_output.joints_from_verts.reshape(inps.shape[0], 17, 3) / 2

        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        pred_uvd_jts = pred_uvd_jts.cpu().data
        pred_mesh = pred_mesh.cpu().data.numpy()
        if test_vertice:
            gt_mesh = gt_mesh.cpu().data.numpy()
            gt_xyz_jts_17 = gt_xyz_jts_17.cpu().data.numpy()

        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(pred_xyz_jts_17.shape[0], 17, 3)
        pred_uvd_jts = pred_uvd_jts.reshape(pred_uvd_jts.shape[0], -1, 3)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(pred_xyz_jts_24.shape[0], 24, 3)
        pred_scores = output.maxvals.cpu().data[:, :29]

        if test_vertice:
            pve = np.sqrt(np.sum((pred_mesh - gt_mesh) ** 2, 2))
            pve_list.append(np.mean(pve) * 1000)

        for i in range(pred_xyz_jts_17.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred_uvd_jts[i], pred_scores[i], hm_shape, bbox, mean_bbox_scale=None)
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17[i],
                'vertices': pred_mesh[i],
                'uvd_jts': pose_coords[0],
                'xyz_24': pred_xyz_jts_24_struct[i]
            }

    with open(os.path.join('exp', f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_pred, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_all_pred = {}
        for r in range(opt.world_size):
            with open(os.path.join('exp', f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join('exp', f'test_gt_kpt_rank_{r}.pkl'))

            kpt_all_pred.update(kpt_pred)

        tot_err_17 = gt_val_dataset.evaluate_xyz_17(kpt_all_pred, os.path.join('exp', 'test_3d_kpt.json'))
        try:
            # _ = gt_val_dataset.evaluate_uvd_24(kpt_all_pred, os.path.join('exp', 'test_3d_kpt.json'))
            _ = gt_val_dataset.evaluate_xyz_24(kpt_all_pred, os.path.join('exp', 'test_3d_kpt.json'))
        except AttributeError:
            pass
        if test_vertice:
            print(f'PVE: {np.mean(pve_list)}')
        return tot_err_17


def main():
    if opt.launcher == 'slurm':
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):

    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    if not opt.log:
        null_writer = NullWriter()
        sys.stdout = null_writer

    torch.backends.cudnn.benchmark = True

    m = builder.build_sppe(cfg.MODEL)

    print(f'Loading model from {opt.checkpoint}...')
    save_dict = torch.load(opt.checkpoint, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        m.load_state_dict(model_dict, strict=False)
    else:
        m.load_state_dict(save_dict, strict=False)

    m.cuda(opt.gpu)
    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[opt.gpu])

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    # gt_val_dataset_hp3d = HP3D(
    #     cfg=cfg,
    #     ann_file='test',
    #     train=False)

    gt_val_dataset_h36m = H36mSMPL(
        cfg=cfg,
        ann_file='Sample_20_test_Human36M_smpl',
        root=cfg.DATASET.DATASET_DIR+'/h36m',
        train=False)

    gt_val_dataset_3dpw = PW3D(
        cfg=cfg,
        ann_file='3DPW_test_new.json',
        root=cfg.DATASET.DATASET_DIR+'/3DPW',
        train=False)

    print('##### Testing on 3DPW #####')
    with torch.no_grad():
        gt_tot_err = validate_gt(m, opt, cfg, gt_val_dataset_3dpw, heatmap_to_coord, opt.batch, test_vertice=True)
    print(f'##### gt 3dpw err: {gt_tot_err} #####')
    # with torch.no_grad():
    #     gt_tot_err = validate_gt(m, opt, cfg, gt_val_dataset_hp3d, heatmap_to_coord, opt.batch)
    # print(f'##### gt 3dhp err: {gt_tot_err} #####')

    print('##### Testing on Human3.6M #####')
    with torch.no_grad():
        gt_tot_err = validate_gt(m, opt, cfg, gt_val_dataset_h36m, heatmap_to_coord, opt.batch)
    print(f'##### gt h36m err: {gt_tot_err} #####')


if __name__ == "__main__":
    main()
