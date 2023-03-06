"""Image demo script."""
import argparse
import os

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from multi_person_tracker import MPT
from torchsummaryX import summary

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    # self.path = input_source
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # bitrate = int(stream.get(cv2.CAP_PROP_BITRATE))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()

    return stream, videoinfo, datalen


def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
# parser.add_argument('--img-path',
#                     help='image name',
#                     default='',
#                     type=str)
parser.add_argument('--video-name',
                    help='video name',
                    default='demo/dance.mp4',
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)
parser.add_argument('--save-pt', default=False, dest='save_pt',
                    help='save prediction', action='store_true')
parser.add_argument('--save-img', default=True, dest='save_img',
                    help='save prediction', action='store_true')

opt = parser.parse_args()


cfg_file = 'configs/256x192_adam_lr1e_3_hrw48_cam_2x_w_pw3d_3dhp_dev_sample_deform_da.yaml'
CKPT = 'pretrained_model/epoch_23_dev_sample_deform_da_h36m_29.83_46.16_3dpw_41.57_73.02.pth'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

res_keys = [
    'pred_uvd',
    'pred_xyz_17',
    'pred_xyz_29',
    'pred_xyz_24_struct',
    'pred_scores',
    'pred_camera',
    'f',
    'pred_betas',
    'pred_thetas',
    'pred_phi',
    'scale_mult',
    'pred_cam_root',
    # 'features',
    'transl',
    'bbox',
    'height',
    'width',
    'img_path'
]
res_db = {k: [] for k in res_keys}

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    flip=True,
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

print('### Extract Image...')
video_basename = os.path.basename(opt.video_name).split('.')[0]

if not os.path.exists(opt.out_dir):
    opt.out_dir = opt.video_name.replace('.mp4', '')
    os.makedirs(opt.out_dir, exist_ok=True)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
# 每帧临时结果
if not os.path.exists(os.path.join(opt.out_dir, 'res_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'))
# 每帧最终结果
if not os.path.exists(os.path.join(opt.out_dir, 'final_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'final_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'final_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'final_2d_images'))

# mp4处理到raw_images

_, info, _ = get_video_info(opt.video_name)
video_basename = os.path.basename(opt.video_name).split('.')[0]

savepath = f'./{opt.out_dir}/res_{video_basename}.mp4'
savepath2d = f'./{opt.out_dir}/res_2d_{video_basename}.mp4'
info['savepath'] = savepath
info['savepath2d'] = savepath2d

write_stream = cv2.VideoWriter(
    *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
write2d_stream = cv2.VideoWriter(
    *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])
if not write_stream.isOpened():
    print("Try to use other video encoders...")
    ext = info['savepath'].split('.')[-1]
    fourcc, _ext = recognize_video_ext(ext)
    info['fourcc'] = fourcc
    info['savepath'] = info['savepath'][:-4] + _ext
    info['savepath2d'] = info['savepath2d'][:-4] + _ext
    write_stream = cv2.VideoWriter(
        *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
    write2d_stream = cv2.VideoWriter(
        *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])

assert write_stream.isOpened(), 'Cannot open video for writing'
assert write2d_stream.isOpened(), 'Cannot open video for writing'

if len(os.listdir(f'{opt.out_dir}/raw_images')) == 0:
    print('视频帧不存在, ffmpeg处理')
    os.system(f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/%06d.jpg')
else:
    print('视频帧存在, 跳过处理')



files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:

        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)


# bbox
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

mot = MPT(
            device=device,
            batch_size=1,
            display=False,
            detector_type='yolo',
            output_format='list',
            yolo_img_size=416,
            )
tracking_results = mot(os.path.join(opt.out_dir, 'raw_images'))
assert len(tracking_results) == len(files), "检测到的帧和图片帧数量不一致"


prev_box = None
renderer = None
smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

print('### Run Model...')
idx = 0
for img_idx in tqdm(range(len(img_path_list))):
    img_path = img_path_list[img_idx]
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    with torch.no_grad():
        # 读取图片
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
  
       # 处理检测框
        image_bboxs = tracking_results[img_idx]

        # 开始检测

        # 画出mesh
        # image = input_image.copy()
        focal = 1000.0
        output_mesh_vis = input_image.copy()
        output_ps_vis = input_image.copy()
        alpha = 0.9  
        for bbox_idx in range(len(image_bboxs)):
            # Run HybrIK
            # image_bboxs: [x1,y1,x2,y2,person_idx] image_bbox: [x1, y1, x2, y2]
            image_bbox = image_bboxs[bbox_idx][:4]
            pose_input, bbox, img_center = transformation.test_transform(
                input_image, image_bbox)
            pose_input = pose_input.to(opt.gpu)[None, :, :, :]


            # 如果测试参数量的话，这里就跑一下就可以
            # summary(hybrik_model,pose_input)
            pose_output = hybrik_model(
                pose_input
                , flip_test=True,
                bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
            )
            uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
            transl = pose_output.transl.detach()

            # Visualization
            # image = input_image.copy()
            # focal = 1000.0
            bbox_xywh = xyxy2xywh(bbox)

            focal_vis = focal / 256 * bbox_xywh[2]

            vertices = pose_output.pred_vertices.detach()

            verts_batch = vertices
            transl_batch = transl

            color_batch = render_mesh(
                vertices=verts_batch, faces=smpl_faces,
                translation=transl_batch,
                focal_length=focal_vis, height=input_image.shape[0], width=input_image.shape[1])  # [1, 540, 960, 4]  4分别是rgb像素和mask

            valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
            image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
            image_vis_batch = (image_vis_batch * 255).cpu().numpy()  # [N,H,W,3] N多少个人

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
            write_stream.write(writer_output_mesh_vis)

            # 保存单个Mesh
            if opt.save_img:
                res_path = os.path.join(opt.out_dir, 'res_images', f'{img_idx:06d}_{bbox_idx}.jpg')
                cv2.imwrite(res_path, writer_output_mesh_vis)

            # vis 2d
            pts = uv_29 * bbox_xywh[2]
            pts[:, 0] = pts[:, 0] + bbox_xywh[0]
            pts[:, 1] = pts[:, 1] + bbox_xywh[1]
            output_ps_vis = vis_2d(output_ps_vis, image_bbox, pts)
            writer_output_ps_vis = cv2.cvtColor(output_ps_vis, cv2.COLOR_RGB2BGR)

            if opt.save_img:
                res_path = os.path.join(
                    opt.out_dir, 'res_2d_images', f'{img_idx:06d}_{bbox_idx}.jpg')
                cv2.imwrite(res_path, writer_output_ps_vis)

        # 图片最终Mesh写入
        write_stream.write(writer_output_mesh_vis)
        # 图片最终2d写入
        write2d_stream.write(writer_output_ps_vis)

        # 保存图片最终mesh结果
        if opt.save_img:
            res_path = os.path.join(opt.out_dir, 'final_images', f'{img_idx:06d}.jpg')
            cv2.imwrite(res_path, writer_output_mesh_vis)

        if opt.save_img:
            res_path = os.path.join(
                opt.out_dir, 'final_2d_images', f'{img_idx:06d}.jpg')
            cv2.imwrite(res_path, writer_output_ps_vis)

            # if opt.save_pt:
            #     assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

            #     pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(
            #         17, 3).cpu().data.numpy()
            #     pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
            #         -1, 3).cpu().data.numpy()
            #     pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(
            #         -1, 3).cpu().data.numpy()
            #     pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(
            #         24, 3).cpu().data.numpy()
            #     pred_scores = pose_output.maxvals.cpu(
            #     ).data[:, :29].reshape(29).numpy()
            #     pred_camera = pose_output.pred_camera.squeeze(
            #         dim=0).cpu().data.numpy()
            #     pred_betas = pose_output.pred_shape.squeeze(
            #         dim=0).cpu().data.numpy()
            #     pred_theta = pose_output.pred_theta_mats.squeeze(
            #         dim=0).cpu().data.numpy()
            #     pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            #     pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            #     img_size = np.array((input_image.shape[0], input_image.shape[1]))

            #     res_db['pred_xyz_17'].append(pred_xyz_jts_17)
            #     res_db['pred_uvd'].append(pred_uvd_jts)
            #     res_db['pred_xyz_29'].append(pred_xyz_jts_29)
            #     res_db['pred_xyz_24_struct'].append(pred_xyz_jts_24_struct)
            #     res_db['pred_scores'].append(pred_scores)
            #     res_db['pred_camera'].append(pred_camera)
            #     res_db['f'].append(1000.0)
            #     res_db['pred_betas'].append(pred_betas)
            #     res_db['pred_thetas'].append(pred_theta)
            #     res_db['pred_phi'].append(pred_phi)
            #     res_db['pred_cam_root'].append(pred_cam_root)
            #     # res_db['features'].append(img_feat)
            #     res_db['transl'].append(transl[0].cpu().data.numpy())
            #     res_db['bbox'].append(np.array(bbox))
            #     res_db['height'].append(img_size[0])
            #     res_db['width'].append(img_size[1])
            #     res_db['img_path'].append(img_path)

        #保存final_mesh
        # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        #     # 保存单个Mesh
        #     if opt.save_img:
        #         idx += 1
        #         res_path = os.path.join(opt.out_dir, 'res_images', f'{idx:06d}_{i}.jpg')
        #         cv2.imwrite(res_path, image_vis)
        #     write_stream.write(image_vis)

write_stream.release()
write2d_stream.release()
# 上面的write写出的视频vscode不能看，手动图片转视频
fina_image_dir = os.path.join(opt.out_dir, 'final_images')
print('fina_image_dir: ' + fina_image_dir)
os.system(f'python /home/ssw/code/tool/tool.py -m i -i {fina_image_dir}')




"""
dev 参数量 [1,3,256,256]
[1,3,256,256] dev
                             Totals
Total params            161.620382M
Trainable params        161.620382M
Non-trainable params            0.0
Mult-Adds             24.762578432G

5人渲染
不渲染
dev [02:22<00:00,  1.17s/it]
63/122 [00:48<00:45,  1.34it/s]

单人
单人只计算不渲染

dev
43/300 [00:22<02:12,  1.94it/s]
97/300 [00:16<00:34,  6.04it/s]

"""

