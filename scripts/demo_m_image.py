"""Image demo script."""
import argparse
import os, sys
import os.path as osp
project_dir = osp.join(osp.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_dir)  # '/home/ssw/code/romp/romp/' + 'lib'

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d

from tqdm import tqdm
from multi_person_tracker import MPT


det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--img',
                    help='image',
                    default='demo/crowd1.jpg',
                    type=str)
parser.add_argument('--img_dir',
                    help='image ',
                    default='demo/demo_origin',
                    type=str)                    
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)
opt = parser.parse_args()


# cfg_file = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml'
# CKPT = './pretrained_w_cam.pth'
cfg_file = 'configs/256x192_adam_lr1e_3_hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = 'pretrained_model/epoch_19_origin_43.07_3dpw_50.61.pth'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    flip = True,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])


hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

hybrik_model.cuda(opt.gpu)
hybrik_model.eval()

if opt.img_dir:
    print('img_dir: ', opt.img_dir)
    img_names = sorted(os.listdir(opt.img_dir))
    img_paths = []
    for img in img_names:
        if '.' in img:
            img_paths.append(os.path.join(opt.img_dir, img))
        else:
            print('not img: ', img)
else:
    print('img_path: ', opt.img)
    img_paths  = [opt.img]


for img in tqdm(img_paths):
    print('img: ', img)
    opt.img = img
    print('### 处理图片目录...')
    if opt.out_dir == '': # output_dir不存在
        opt.out_dir = osp.join(osp.dirname(opt.img), 'output')
        out_tmp_video_dir = osp.join(opt.out_dir, 'tmp_video', osp.basename(opt.img).split('.')[0])
    else:
        out_tmp_video_dir = osp.join(opt.out_dir, 'tmp_video', osp.basename(opt.img).split('.')[0])

    os.makedirs(opt.out_dir, exist_ok=True)
    os.makedirs(opt.out_dir + '/tmp', exist_ok=True)
    print('output dir: ' + opt.out_dir)
    os.makedirs(out_tmp_video_dir, exist_ok=True)
    os.system(f'cp {opt.img} {out_tmp_video_dir}')


    files = os.listdir(osp.join(project_dir, out_tmp_video_dir))
    smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

    # 定义得到bbox
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mot = MPT(
                device=device,
                batch_size=1,
                display=False,
                detector_type='yolo',
                output_format='list',
                yolo_img_size=416,
                )
    tracking_results = mot(out_tmp_video_dir)
    # 开始处理图片


    # process file name
    img_path = opt.img
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    # 得到bbox
    input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # [375, 500, 3]
    image_bboxs = tracking_results[0]


    # 画出mesh
    # image = input_image.copy()
    focal = 1000.0
    output_mesh_vis = input_image.copy()
    output_ps_vis = input_image.copy()
    alpha = 1  # 原人体显示占比多少
    i = 0 
    for bbox_idx in range(len(image_bboxs)):  # image_bboxs [11,5]
        # output_mesh_vis = input_image.copy()  # 每次在原图生成一个mesh,结果不重叠

        i += 1
        # Run HybrIK
        # image_bboxs: [x1,y1,x2,y2,person_idx] image_bbox: [x1, y1, x2, y2]
        image_bbox = image_bboxs[bbox_idx][:4]
        pose_input, bbox, img_center = transformation.test_transform(
            input_image, image_bbox)  # pose_input [1,3,256,256]由处理后的Bbox变成, bbox处理后的Bbox,长宽相同, img_center原始图像的
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]  # [1, 3, 256, 256]
        pose_output = hybrik_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
        )
        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        transl = pose_output.transl.detach()  # [ 8.6759, -2.1994,  7.9684]

        # Visualization
        # image = input_image.copy()
        # focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)  # [471.05, 134.11, 51.55, 51.55]<- [445.27, 108.33, 496.83, 159.89]

        focal_vis = focal / 256 * bbox_xywh[2]  # 201.38

        vertices = pose_output.pred_vertices.detach()

        verts_batch = vertices
        transl_batch = transl

        color_batch = render_mesh(
            vertices=verts_batch, faces=smpl_faces,
            translation=transl_batch,
            focal_length=focal_vis, height=input_image.shape[0], width=input_image.shape[1])  # [1, 540, 960, 4]  4分别是rgb像素和mask

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0) # 有图案的mask [1, 375, 500, 4]
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch  # [1, 375, 500, 4] 画出图案
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()  # [N,H,W,3] N多少个人 转换成rgb

        # # 画出mesh 
        # output_mesh_vis = image
        # alpha = 0.9  

        color = image_vis_batch[0] # 540, 960, 3
        valid_mask = valid_mask_batch[0].cpu().numpy()  # 540, 960, 1

        # alpha * color[:, :, :3] * valid_mask画出人体位置mesh部分，占比0.9,  ( 1 - alpha) * output_mesh_vis * valid_mask,人体位置人体部分  (1 - valid_mask) * output_mesh_vis人体之外的部分
        output_mesh_vis = alpha * color[:, :, :3] * valid_mask + (
            1 - alpha) * output_mesh_vis * valid_mask + (1 - valid_mask) * output_mesh_vis

        output_mesh_vis = output_mesh_vis.astype(np.uint8)
        writer_output_mesh_vis = cv2.cvtColor(output_mesh_vis, cv2.COLOR_RGB2BGR)

        # 保存单个Mesh
        # if opt.save_img:
        res_path = os.path.join(opt.out_dir, 'tmp', basename.split('.')[0] + f'_{i}.jpg')
        cv2.imwrite(res_path, writer_output_mesh_vis)

        # vis 2d
        pts = uv_29 * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]
        output_ps_vis = vis_2d(output_ps_vis, image_bbox, pts)
        writer_output_ps_vis = cv2.cvtColor(output_ps_vis, cv2.COLOR_RGB2BGR)

    # 保存图片最终mesh结果
    res_path = os.path.join(opt.out_dir, basename.split('.')[0] + '_mesh.jpg')
    cv2.imwrite(res_path, writer_output_mesh_vis)

    res_path = os.path.join(opt.out_dir, basename.split('.')[0] + '_2d.jpg')
    cv2.imwrite(res_path, writer_output_ps_vis)
        # if opt.save_img:
        #     res_path = os.path.join(
        #         opt.out_dir, 'res_2d_images', f'{img_idx:06d}_{bbox_idx}.jpg')
        #     cv2.imwrite(res_path, writer_output_ps_vis)