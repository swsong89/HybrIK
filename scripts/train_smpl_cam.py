"""Script for multi-gpu training."""
import argparse
import logging
import os
import pickle as pk
import random
import sys
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data
from torch.nn.utils import clip_grad

from hybrik.datasets import MixDataset, MixDatasetCam, PW3D, MixDataset2Cam
from hybrik.models import builder
from hybrik.opt import cfg, logger, opt
from hybrik.utils.env import init_dist
from hybrik.utils.metrics import DataLogger, NullWriter, calc_coord_accuracy
from hybrik.utils.transforms import get_func_heatmap_to_coord, torch_std_to_img, torch_to_im, cv_cropBoxInverse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d, xyxy2xywh
from hybrik.utils.render_pytorch3d import render_mesh



# torch.set_num_threads(64)
num_gpu = torch.cuda.device_count()
valid_batch = 1 * num_gpu


def _init_fn(worker_id):
    np.random.seed(opt.seed + worker_id)
    random.seed(opt.seed + worker_id)

smpl_faces = None

def train(m, opt, train_loader, criterion, optimizer, writer, epoch, cfg, gt_val_dataset_3dpw, heatmap_to_coord):
    print('bathc_size: ', cfg.TRAIN.get('BATCH_SIZE'))  # cfg.TRAIN.BATCH_SIZE如果不存在会报错, cfg.TRAIN.get('BATCH_SIZE')不存在返回None
    loss_logger = DataLogger()
    acc_uvd_29_logger = DataLogger()
    acc_xyz_17_logger = DataLogger()
    m.train()
    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    depth_dim = cfg.MODEL.EXTRA.get('DEPTH_DIM')
    hm_shape = (hm_shape[1], hm_shape[0], depth_dim)
    root_idx_17 = train_loader.dataset.root_idx_17

    # if opt.log:
    #     train_loader = tqdm(train_loader, dynamic_ncols=True)
    epoch_start_time = time.time()
    iter_start_time = time.time()

    iters = len(train_loader)
    test_internal_iters = iters//20
    print('test_internal_iters: ', test_internal_iters)

    for iter, (inps, labels, idxs, img_paths, bboxes) in enumerate(train_loader):

        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu).requires_grad_() for inp in inps]
        else:
            inps = inps.cuda(opt.gpu).requires_grad_()

        for k, _ in labels.items():
            labels[k] = labels[k].cuda(opt.gpu)

        trans_inv = labels.pop('trans_inv')
        intrinsic_param = labels.pop('intrinsic_param')
        root = labels.pop('joint_root')
        depth_factor = labels.pop('depth_factor')

        output = m(inps, trans_inv=trans_inv, intrinsic_param=intrinsic_param, joint_root=root, depth_factor=depth_factor)
        # 处理后的图片网络是256,256
        transofr_img_height = 256
        vis_ratio = 4
        transofr_img_vis_height = transofr_img_height*vis_ratio
        bboxes = [0, transofr_img_vis_height, transofr_img_vis_height, 0]  # 左下角 右上角 左上角原点，往右x轴，往下y轴
        if opt.debug:
            for i in range(len(img_paths)):
                # transofr_img vis
                img = inps[i].detach().cpu()
                transfor_img = torch_std_to_img(img)
                transfor_img = cv2.normalize(transfor_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # 结果是浮点数，可能是小数，imshow会自动标准化，imwrite不会，以防万一手动进行
                transfor_img_vis = cv2.cvtColor(transfor_img, cv2.COLOR_RGB2BGR)
                transfor_img_vis = cv2.resize(transfor_img_vis,(bboxes[1], bboxes[1]),interpolation=cv2.INTER_CUBIC)   #dsize=（2*width,2*height）



                # # # origin img vis
                # origin_img= cv2.cvtColor(cv2.imread(img_paths[i]), cv2.COLOR_BGR2RGB)
                # origin_img_vis = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)


                #origin_bbox_img
                transfor_img_double = cv2.resize(transfor_img,(bboxes[1], bboxes[1]),interpolation=cv2.INTER_CUBIC)   #dsize=（2*width,2*height）

                bbox_xywh = xyxy2xywh(bboxes)  # 580.9016, 1105.1338, 1609.7358, 1609.7358 <- [-223.9663,  300.2659, 1385.7695, 1910.0017]
                origin_uv_29 = labels['target_uvd_29'][i].detach().reshape(29, 3)[:, :2]
                origin_pts = origin_uv_29 * bbox_xywh[2]
                origin_pts[:, 0] = origin_pts[:, 0] + bbox_xywh[0]
                origin_pts[:, 1] = origin_pts[:, 1] + bbox_xywh[1]
                origin_bbox_img = vis_2d(transfor_img_double, bboxes, origin_pts)
                orgin_bbox_img_vis = cv2.cvtColor(origin_bbox_img, cv2.COLOR_RGB2BGR)


                #pred_bbox_img
                uv_29 = output.pred_uvd_jts[i].detach().reshape(29, 3)[:, :2]
                pts = uv_29 * bbox_xywh[2]
                pts[:, 0] = pts[:, 0] + bbox_xywh[0]
                pts[:, 1] = pts[:, 1] + bbox_xywh[1]
                pred_bbox_img = vis_2d(transfor_img_double, bboxes, pts)
                pred_bbox_img = cv2.cvtColor(pred_bbox_img, cv2.COLOR_RGB2BGR)


                # origin mesh
                # focal = 1000.0
                # focal = focal/256*bbox_xywh[3]
                # transl = output.transl.detach()

                # vertices = output.pred_vertices.detach()

                # verts_batch = vertices
                # transl_batch = transl

                # color_batch = render_mesh(
                #     vertices=verts_batch, faces=smpl_faces,
                #     translation=transl_batch,
                #     focal_length=focal, height=origin_img.shape[0], width=origin_img.shape[1])

                # valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
                # mesh_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
                # mesh_vis_batch = (mesh_vis_batch * 255).cpu().numpy()

                # color = mesh_vis_batch[0]
                # valid_mask = valid_mask_batch[0].cpu().numpy()
                # alpha = 0.9
                # mesh_img = alpha * color[:, :, :3] * valid_mask + (
                #     1 - alpha) * origin_img * valid_mask + (1 - valid_mask) * origin_img

                # mesh_img = mesh_img.astype(np.uint8)
                # mesh_img_vis = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)


                # transofr mesh 256
                focal = 1000.0
                transl = output.transl[None,i].detach()

                vertices = output.pred_vertices[None,i].detach()

                verts_batch = vertices
                transl_batch = transl
                color_batch = render_mesh(
                    vertices=verts_batch, faces=smpl_faces,
                    translation=transl_batch,
                    focal_length=focal, height=transfor_img.shape[0], width=transfor_img.shape[1])

                valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
                mesh_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
                mesh_vis_batch = (mesh_vis_batch * 255).cpu().numpy()

                color = mesh_vis_batch[0]
                valid_mask = valid_mask_batch[0].cpu().numpy()
                alpha = 0.9
                mesh_img = alpha * color[:, :, :3] * valid_mask + (
                    1 - alpha) * transfor_img * valid_mask + (1 - valid_mask) * transfor_img

                mesh_img = mesh_img.astype(np.uint8)
                transfor_mesh_img_vis = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)
                transfor_mesh_img_vis = cv2.resize(transfor_mesh_img_vis,(bboxes[1], bboxes[1]),interpolation=cv2.INTER_CUBIC)   #dsize=（2*width,2*height）



                origin_pred_bbox_img = np.hstack((orgin_bbox_img_vis, pred_bbox_img))
                # img_and_img_mesh = np.hstack((origin_img_vis, mesh_img_vis))
                transfor_and_mesh_img = np.hstack((transfor_img_vis, transfor_mesh_img_vis))

                # transfor transfor_mesh origin_bbox pred_bbox
                transfor_vis_all = np.vstack((transfor_and_mesh_img, origin_pred_bbox_img))
                if opt.show:
                    # bbox
                    # cv2.imshow('origin_pred_bbox_img', origin_pred_bbox_img)

                    # cv2.imshow('orgin_bbox_img_vis', orgin_bbox_img_vis)
                    # cv2.imshow('pred_bbox_img', bbox_vis)

                    # mesh   
                    # cv2.imshow('img_and_img_mesh', img_and_img_mesh)

                    # cv2.imshow('origin_img', origin_img_vis)
                    # cv2.imshow('mesh_img_origin_vis', mesh_img_vis)

                    # transofor_img
                    # cv2.imshow('transfor_and_mesh_img', transfor_and_mesh_img)

                    # cv2.imshow('transfor_img_vis', transfor_img_vis)
                    # cv2.imshow('transfor_mesh_img_vis', transfor_mesh_img_vis)

                    # transfor vis all
                    cv2.imshow('transfor_mesh_origin_pred', transfor_vis_all)
                    

                    key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭
                    cv2.destroyAllWindows()
                # else:
                # bbox 
                # origin_pred_bbox_img = np.hstack((orgin_bbox_img_vis, bbox_vis))
                # cv2.imwrite('configs/tmp/bbox_origin_and_pred.jpg', origin_pred_bbox_img)

                # mesh 
                # img_and_img_mesh = np.hstack((origin_img_vis, mesh_img_vis))
                # cv2.imwrite('configs/tmp/mesh_and_origin.jpg', img_and_img_mesh)

                # transofor_img
                # transfor_and_mesh_img = np.hstack((transfor_img_vis, transfor_mesh_img_vis))
                # cv2.imwrite('configs/tmp/transfor_and_pred.jpg', transfor_and_mesh_img)

                cv2.imwrite('configs/tmp/transfor_mesh_origin_pred.jpg', transfor_vis_all)
                # print('ok')

        loss = criterion(output, labels)

        pred_uvd_jts = output.pred_uvd_jts
        pred_xyz_jts_17 = output.pred_xyz_jts_17
        label_masks_29 = labels['target_weight_29']
        label_masks_17 = labels['target_weight_17']

        if pred_uvd_jts.shape[1] == 24 or pred_uvd_jts.shape[1] == 72:
            pred_uvd_jts = pred_uvd_jts.cpu().reshape(pred_uvd_jts.shape[0], 24, 3)
            gt_uvd_jts = labels['target_uvd_29'].cpu().reshape(pred_uvd_jts.shape[0], 29, 3)[:, :24, :]
            gt_uvd_mask = label_masks_29.cpu().reshape(pred_uvd_jts.shape[0], 29, 3)[:, :24, :]
            acc_uvd_29 = calc_coord_accuracy(pred_uvd_jts, gt_uvd_jts, gt_uvd_mask, hm_shape, num_joints=24)
        else:
            acc_uvd_29 = calc_coord_accuracy(pred_uvd_jts.detach().cpu(), labels['target_uvd_29'].cpu(), label_masks_29.cpu(), hm_shape, num_joints=29)
        acc_xyz_17 = calc_coord_accuracy(pred_xyz_jts_17.detach().cpu(), labels['target_xyz_17'].cpu(), label_masks_17.cpu(), hm_shape, num_joints=17, root_idx=root_idx_17)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_uvd_29_logger.update(acc_uvd_29, batch_size)
        acc_xyz_17_logger.update(acc_xyz_17, batch_size)

        optimizer.zero_grad()
        loss.backward()

        for group in optimizer.param_groups:
            for param in group["params"]:
                clip_grad.clip_grad_norm_(param, 5)

        optimizer.step()

        if opt.log:
            # TQDM 下面set_description是给tqdm显示打印的，变成loss: 8.06730000 | accuvd29: 0.0813 | acc17: 0.0800:   0%|▏                                                          | 254/78047 [00:40<3:26:42,  6.27it/s]
            # loss desciption loss: 7.54187632 | accuvd29: 0.0575 | acc17: 0.0000
            loss_desciption ='loss: {loss:.4f} | accuvd29: {accuvd29:.4f} | acc17: {acc17:.4f}'.format(
                    loss=loss_logger.avg,
                    accuvd29=acc_uvd_29_logger.avg,
                    acc17=acc_xyz_17_logger.avg)
            # train_loader.set_description(loss_desciption)
            # logging的频率
            
            if iter % opt.print_freq == 0:
                time_print_freq = time.time() - iter_start_time  # 计算迭代打印频率时间
                iter_start_time = time.time()
                pred_time = len(train_loader)/opt.print_freq*time_print_freq/60  #预测整个epoch需要时间
                epoch_used_time = (time.time() - epoch_start_time)/60  # epoch已经使用的时间
                loss_desciption += '  time: {:.2f}s  pred_time: {:.2f}m used_time: {:.2f}m'.format(
                                    time_print_freq, pred_time, epoch_used_time)
                logger.iterInfo(epoch, iter, len(train_loader), loss_desciption)

        if (iter+1) % test_internal_iters == 0: #保存的间隔  # iter从0开始，避免刚开始就保存模型
            time_str = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
            cache_model_path =  opt.work_dir + '/checkpoint/cache_model.pth'
            logger.info('=====> ' + time_str + ' Saving epoch_{}_iter_{} cache_model_path: '.format(epoch, iter) + cache_model_path)
            torch.save(m.module.state_dict(), cache_model_path)
            # Prediction Test
            # logger.info('=>  Start validate 3DPW dataset,  cache_model_path: ' + cache_model_path)
            # with torch.no_grad():
            #     gt_tot_err_3dpw = validate_gt(m, opt, cfg, gt_val_dataset_3dpw, heatmap_to_coord)
            #     logger.info('=> epoch {} iter {} cache model 3dpw error {:.2f}'.format(epoch, iter, gt_tot_err_3dpw))
                # Save val checkpoint
                # torch.save(m.module.state_dict(), opt.work_dir + '/checkpoint/epoch_{}_3dpw_{:.2f}.pth'.format(epoch, gt_tot_err_3dpw))
    


    return loss_logger.avg, acc_xyz_17_logger.avg


def validate_gt(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=cfg.TRAIN.get('BATCH_SIZE')//2, pred_root=False):  # 3/4 batchsize在h36m和pw3d的情况下不会超出训练需要的内存
    print('batch_size: ', batch_size)

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, sampler=gt_val_sampler, pin_memory=True)
    kpt_pred = {}
    m.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    if opt.tqdm:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    for val_iter, (inps, labels, img_ids, bboxes) in enumerate(gt_val_loader):
        if val_iter % 500 == 0:
            print('{} /{}'.format(val_iter, len(gt_val_loader)))
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].cuda(opt.gpu)
            except AttributeError:
                assert k == 'type'

        output = m(inps, flip_test=opt.flip_test, bboxes=bboxes,
                   img_center=labels['img_center'])

        # pred_xyz_jts_29 = output.pred_xyz_jts_29.reshape(inps.shape[0], -1, 3)
        pred_xyz_jts_24 = output.pred_xyz_jts_29.reshape(inps.shape[0], -1, 3)[:, :24, :]
        pred_xyz_jts_24_struct = output.pred_xyz_jts_24_struct.reshape(inps.shape[0], 24, 3)
        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)

        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()

        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(
            pred_xyz_jts_17.shape[0], 17, 3)
        # pred_uvd_jts = pred_uvd_jts.reshape(
        #     pred_uvd_jts.shape[0], -1, 3)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(
            pred_xyz_jts_24.shape[0], 24, 3)
        # pred_scores = output.maxvals.cpu().data[:, :29]

        for i in range(pred_xyz_jts_17.shape[0]):
            # bbox = bboxes[i].tolist()
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17[i],
                'xyz_24': pred_xyz_jts_24[i]
            }

    with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_pred, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_all_pred = {}
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'))

            kpt_all_pred.update(kpt_pred)

        tot_err_17 = gt_val_dataset.evaluate_xyz_17(
            kpt_all_pred, os.path.join(opt.work_dir, 'test_3d_kpt.json'))

        return tot_err_17


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    # print(opt)
    # opt = argparse.Namespace(board=True, cfg='configs/256x192_adam_lr1e_3_res34_smpl_3d_cam_2x_mix_w_pw3d.yaml', debug=False, dist_backend='nccl', dist_url='tcp://127.0.1.1:23456', dynamic_lr=False, exp_id='test_3dpw', exp_lr=False, flip_shift=False, flip_test=True, launcher='pytorch', map=True, nThreads=8, params=False, rank=0, seed=123123, snapshot=2, sync=False, work_dir='./exp/mix2_smpl_cam/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml-test_3dpw/', world_size=0)
    # print(opt)
    if opt.seed is not None:
        setup_seed(opt.seed)

    if opt.launcher == 'slurm':
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))
        main_worker(0, opt, cfg)


def main_worker(gpu, opt, cfg):
    if opt.seed is not None:
        setup_seed(opt.seed)

    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    if not opt.log:
        logger.setLevel(50)
        null_writer = NullWriter()
        sys.stdout = null_writer

    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')


    # 处理下pretrained问题
    try:
        hr_pretrained_model_name = cfg.MODEL.HR_PRETRAINED
    except Exception as e:
        cfg.MODEL.HR_PRETRAINED = ''
        print('hr pretrained is not exist')
    try:
        pretrained_model_name = cfg.MODEL.PRETRAINED
    except Exception as e:
        cfg.MODEL.PRETRAINED = ''
        print('pretrained is not exist')
    try:
        try_load = cfg.MODEL.TRY_LOAD
    except Exception as e:
        cfg.MODEL.TRY_LOAD = ''
        print('try load is not exist')

    # continue train
    if opt.ct:
        checkpoint_path = os.path.join(opt.work_dir, 'checkpoint')
        files = os.listdir(checkpoint_path)
        try:
            max_checkpoint_path = max(files)
            begin_epoch = int(max_checkpoint_path.split('/')[-1].split('_')[1]) + 1  # epoch_1_iter_2,从epoch下一个开始
            cfg.TRAIN.BEGIN_EPOCH = begin_epoch
            cfg.MODEL.PRETRAINED = checkpoint_path + '/' + max_checkpoint_path
            logger.info('Find newest checkpoint_path: ' + cfg.MODEL.PRETRAINED)

        except Exception as e:
            # logger.info('max_checkpoint_path: {}, unvalid'.format(checkpoint_path + '/' + max_checkpoint_path))
            logger.info('New train begin epoch : ' + str(cfg.TRAIN.BEGIN_EPOCH) + ', pretrained_model path: ' + cfg.MODEL.PRETRAINED)

    opt.nThreads = int(opt.nThreads / num_gpu)

    # Model Initialize
    m = preset_model(cfg)
    if opt.params:
        from thop import clever_format, profile
        input = torch.randn(1, 3, 256, 256).cuda(opt.gpu)
        flops, params = profile(m.cuda(opt.gpu), inputs=(input, ))
        macs, params = clever_format([flops, params], "%.3f")
        logger.info(macs, params)

    m.cuda(opt.gpu)
    global smpl_faces
    smpl_faces = torch.from_numpy(m.smpl.faces.astype(np.int32))
    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[opt.gpu])

    criterion = builder.build_loss(cfg.LOSS).cuda(opt.gpu)
    optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    if opt.log:
        writer = SummaryWriter('.tensorboard/{}/{}-{}'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))
    else:
        writer = None

    if cfg.DATASET.DATASET == 'mix_smpl':
        train_dataset = MixDataset(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == 'mix_smpl_cam':
        train_dataset = MixDatasetCam(
            cfg=cfg,
            train=True)
    elif cfg.DATASET.DATASET == 'mix2_smpl_cam':
        train_dataset = MixDataset2Cam(
            cfg=cfg,
            train=True)
    else:
        raise NotImplementedError

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=opt.world_size, rank=opt.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), num_workers=opt.nThreads, sampler=train_sampler, worker_init_fn=_init_fn, pin_memory=True)

    # gt val dataset
    if cfg.DATASET.DATASET == 'mix_smpl':
        gt_val_dataset_h36m = MixDataset(
            cfg=cfg,
            train=False)
    elif cfg.DATASET.DATASET == 'mix_smpl_cam' or cfg.DATASET.DATASET == 'mix2_smpl_cam':
        gt_val_dataset_h36m = MixDatasetCam(
            cfg=cfg,
            train=False)
        print('valid dataset h36m_len: {}'.format(len(gt_val_dataset_h36m)))
    else:
        raise NotImplementedError

    gt_val_dataset_3dpw = PW3D(
        cfg=cfg,
        ann_file='3DPW_test_new.json',
        root=cfg.DATASET.DATASET_DIR+'/3DPW',
        train=False)
    print('valid dataset 3dpw_len: {}'.format(len(gt_val_dataset_3dpw)))


    best_err_h36m = 999
    best_err_3dpw = 999

    if opt.fast_eval:
        print('-----------fast eval----------------')
        with torch.no_grad():
            gt_tot_err_h36m = validate_gt(m, opt, cfg, gt_val_dataset_h36m, heatmap_to_coord)
            gt_tot_err_3dpw = validate_gt(m, opt, cfg, gt_val_dataset_3dpw, heatmap_to_coord)

            # Save val checkpoint
            val_checkpoint_path = '/checkpoint/epoch_{}_z_h36m_{:.2f}_3dpw_{:.2f}.pth'.format(cfg.TRAIN.BEGIN_EPOCH, gt_tot_err_h36m, gt_tot_err_3dpw)
            torch.save(m.module.state_dict(), opt.work_dir + val_checkpoint_path)
            logger.info('=>  Saveing val_checkpoint_path: ' + val_checkpoint_path)
            logger.info(f'##### Epoch {cfg.TRAIN.BEGIN_EPOCH} | h36m err: {gt_tot_err_h36m} / {best_err_h36m} | 3dpw err: {gt_tot_err_3dpw} / {best_err_3dpw} #####')

    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        train_sampler.set_epoch(epoch)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {epoch} | LR: {current_lr} #############')

        # Training
        loss, acc17 = train(m, opt, train_loader, criterion, optimizer, writer, epoch,
                            cfg, gt_val_dataset_3dpw, heatmap_to_coord)  # cfg, gt_val_dataset_h36m, heatmap_to_coord 为了test_internal添加的
        logger.epochInfo('Train', epoch, loss, acc17)

        lr_scheduler.step()
        # 每个epoch结束保存一下
        time_str = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
        cache_model_path =  opt.work_dir + '/checkpoint/cache_model.pth'
        logger.info('=====> ' + time_str + ' Saving epoch_{} cache_model_path: '.format(epoch) + cache_model_path)
        torch.save(m.module.state_dict(), cache_model_path)
        if (epoch + 1) % opt.snapshot == 0:
            # if opt.log:
            #     # Save checkpoint
            #     torch.save(m.module.state_dict(), './exp/{}/{}-{}/model_{}.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id, opt.epoch))

            # Prediction Test
            with torch.no_grad():
                gt_tot_err_h36m = validate_gt(m, opt, cfg, gt_val_dataset_h36m, heatmap_to_coord)
                gt_tot_err_3dpw = validate_gt(m, opt, cfg, gt_val_dataset_3dpw, heatmap_to_coord)

                # Save val checkpoint
                val_checkpoint_path = '/checkpoint/epoch_{}_z_h36m_{:.2f}_3dpw_{:.2f}.pth'.format(epoch, gt_tot_err_h36m, gt_tot_err_3dpw)
                torch.save(m.module.state_dict(), opt.work_dir + val_checkpoint_path)
                logger.info('=>  Saveing val_checkpoint_path: ' + val_checkpoint_path)
                if opt.log:
                    if gt_tot_err_h36m <= best_err_h36m:
                        best_err_h36m = gt_tot_err_h36m
                        torch.save(m.module.state_dict(), opt.work_dir + '/best_h36m_model.pth')
                    if gt_tot_err_3dpw <= best_err_3dpw:
                        best_err_3dpw = gt_tot_err_3dpw
                        torch.save(m.module.state_dict(), opt.work_dir + '/best_3dpw_model.pth')

                    logger.info(f'##### Epoch {epoch} | h36m err: {gt_tot_err_h36m} / {best_err_h36m} | 3dpw err: {gt_tot_err_3dpw} / {best_err_3dpw} #####')

        torch.distributed.barrier()  # Sync

    torch.save(m.module.state_dict(), opt.work_dir + '/final_DPG.pth')


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL)

    if cfg.MODEL.PRETRAINED:  # PRETRAINED相当于在这个模型的基础上再进行训练,而不是backbone pretrained_model
        logger.info(f'Loading pretrained model from path: {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED, map_location='cpu'))
    elif cfg.MODEL.TRY_LOAD:  # try load就相当于是预训练加载,比如backbone是hr32,然后部分模块和backbone不一样,那么就加载部分共同的参数
        logger.info(f'Loading try load model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD, map_location='cpu')
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == "__main__":
    main()
