from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import SPPE
from .layers.smpl.SMPL import SMPL_layer
from .layers.hrnet.hrnet import get_hrnet


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def norm_heatmap(norm_type, heatmap, tau=5, sample_num=1):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape  # [1,29,262144]
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)  # [1,29,262144]
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == 'sampling':
        heatmap = heatmap.reshape(*shape[:2], -1)

        eps = torch.rand_like(heatmap)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau

        gumbel_heatmap = F.softmax(gumbel_heatmap, 2)
        return gumbel_heatmap.reshape(*shape)
    elif norm_type == 'multiple_sampling':

        heatmap = heatmap.reshape(*shape[:2], 1, -1)

        eps = torch.rand(*heatmap.shape[:2], sample_num, heatmap.shape[3], device=heatmap.device)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau
        gumbel_heatmap = F.softmax(gumbel_heatmap, 3)
        gumbel_heatmap = gumbel_heatmap.reshape(shape[0], shape[1], sample_num, shape[2])

        # [B, S, K, -1]
        return gumbel_heatmap.transpose(1, 2)
    else:
        raise NotImplementedError

# class GraphConvBlock(nn.Module):
#     def __init__(self, adj, dim_in, dim_out):  # adj=15, dim_in=2052, dim_out=128
#         super(GraphConvBlock, self).__init__()
#         self.adj = adj  # [15,15]
#         self.vertex_num = adj.shape[0]  # [15]
#         self.fcbn_list = nn.ModuleList([nn.Sequential(*[nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out)]) for _ in range(self.vertex_num)])

#     def forward(self, feat):  # [1,15,2052]
#         batch_size = feat.shape[0]  # 1

#         # apply kernel for each vertex, 相当于取每个关节点[1,2048]进行一个卷积得到[1,128]
#         feat = torch.stack([fcbn(feat[:,i,:]) for i,fcbn in enumerate(self.fcbn_list)],1)  # [1,15,128]= 15个[1,1,128] <- [1,1,2052] 卷积和[128,5012]，一个节点不同维度特征之间的信息加权和，然后需要多少维度做多少次。一层神经元数等于一层特征通道数，每个神经元特征就是HW，

#         # apply adj
#         adj = self.adj.cuda()[None, :, :].repeat(batch_size, 1, 1)  # TODO repeat函数是复制，【15，15】 1，1 ->【15，15】 1,2 -> [15,30]
#         feat = torch.bmm(adj, feat)  # 【1，15，128】 <- [15,15], [1,15,128] 不同节点同纬度特征进行信息交换

#         # apply activation function
#         out = F.relu(feat)
#         return out  # 【1，15，1288】


# class GraphResBlock(nn.Module):
#     def __init__(self, adj, dim):
#         super(GraphResBlock, self).__init__()
#         self.adj = adj  # 15
#         self.graph_block1 = GraphConvBlock(adj, dim, dim)  # 15，128，128
#         self.graph_block2 = GraphConvBlock(adj, dim, dim)

#     def forward(self, feat):
#         feat_out = self.graph_block1(feat)
#         feat_out = self.graph_block2(feat_out)
#         out = feat_out + feat
#         return out


@SPPE.register_module
class HRNetSMPLCam(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HRNetSMPLCam, self).__init__()
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        self.pretrain_hrnet = kwargs['HR_PRETRAINED']

        self.preact = get_hrnet(kwargs['HRNET_TYPE'], num_joints=self.num_joints,
                                depth_dim=self.depth_dim,
                                is_train=True, generate_feat=True, generate_hm=True)

        # # Load pretrain model
        # model_state = self.preact.state_dict()
        # state = {k: v for k, v in x.state_dict().items()
        #          if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        # model_state.update(state)
        # self.preact.load_state_dict(model_state)

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype
        )

        self.joint_pairs_24 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

        self.joint_pairs_29 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                               (22, 23), (25, 26), (27, 28))

        self.root_idx_smpl = 0

        # mean shape
        self.joint_num = 29  # 15
        # self.graph_adj = torch.from_numpy(self.smpl.graph_adj).float()  # 15x15

        # # graph convs  2048为C',即F'特征2048x8x8 4=3+1为P 3D +置信度
        # self.graph_block = nn.Sequential(*[\
        #     GraphConvBlock(self.graph_adj, 2048+4, 128),
        #     GraphResBlock(self.graph_adj, 128),
        #     GraphResBlock(self.graph_adj, 128),
        #     GraphResBlock(self.graph_adj, 128),
        #     GraphResBlock(self.graph_adj, 128)])


        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer(
            'init_shape',
            torch.Tensor(init_shape).float())

        init_cam = torch.tensor([0.9])
        self.register_buffer(
            'init_cam',
            torch.Tensor(init_cam).float())
        self.dejoint = nn.Linear(self.joint_num*(2048+4), 1024)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(1024, 1)
        self.decsigma = nn.Linear(1024, 29)

        self.focal_length = kwargs['FOCAL_LENGTH']
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        self.input_size = 256.0

    def _initialize(self):
        self.preact.init_weights(self.pretrain_hrnet)

    def flip_xyz_coord(self, pred_jts, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        pred_jts[:, :, 0] = - pred_jts[:, :, 0]

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.num_joints * 3)

        return pred_jts

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def flip_sigma(self, pred_sigma):

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_sigma[:, idx] = pred_sigma[:, inv_idx]

        return pred_sigma

    def flip_heatmap(self, heatmaps, shift=True):
        heatmaps = heatmaps.flip(dims=(4,))

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            heatmaps[:, idx] = heatmaps[:, inv_idx]

        if shift:
            if heatmaps.dim() == 3:
                heatmaps[:, :, 1:] = heatmaps[:, :, 0:-1]
            elif heatmaps.dim() == 4:
                heatmaps[:, :, :, 1:] = heatmaps[:, :, :, 0:-1]
            else:
                heatmaps[:, :, :, :, 1:] = heatmaps[:, :, :, :, 0:-1]

        return heatmaps

    def forward(self, x, flip_test=False, **kwargs):  # x [1,256,256,3] flip_test=True, bbox [1,4] img_center [1,2]
        batch_size = x.shape[0]
        # 关节点部分
        # x0 = self.preact(x)
        out, x0 = self.preact(x)  # [1, 1856, 64, 64] [1, 2048]
        # print(out.shape)
        out = out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)  # [1, 29, 64, 64, 64]

        if flip_test:
            flip_x = flip(x)  # 图片左右翻转[-0.9678, -0.9853, -1.0027,  ..., -1.3687, -1.4036, -1.4384] <- [-1.4384, -1.4036, -1.3687,  ..., -1.0027, -0.9853, -0.9678]
            flip_out, flip_x0 = self.preact(flip_x)

            # flip heatmap
            flip_out = flip_out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)  # [1, 29, 64, 64, 64]
            flip_out = self.flip_heatmap(flip_out)

            out = out.reshape((out.shape[0], self.num_joints, -1))  # [1, 29, 262144]
            flip_out = flip_out.reshape((flip_out.shape[0], self.num_joints, -1))

            heatmaps = norm_heatmap(self.norm_type, out)
            flip_heatmaps = norm_heatmap(self.norm_type, flip_out)
            heatmaps = (heatmaps + flip_heatmaps) / 2

        else:
            out = out.reshape((out.shape[0], self.num_joints, -1))

            heatmaps = norm_heatmap(self.norm_type, out)

        assert heatmaps.dim() == 3, heatmaps.shape
        # assert hypo_heatmaps.dim() == 4, heatmaps.shape
        # print(hypo_heatmaps.shape)

        maxvals, _ = torch.max(heatmaps, dim=2, keepdim=True)  # [1, 29, 1]  _ [1,29,1]是262144的序号

        # print(out.sum(dim=2, keepdim=True))
        # heatmaps = out / out.sum(dim=2, keepdim=True)

        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))  # [1, 29, 64, 64, 64]

        hm_x0 = heatmaps.sum((2, 3))  # (B, K, W)  # heatmaps 【B，K，D，H，W】
        hm_y0 = heatmaps.sum((2, 4))  # (B, K, H)  # hm_zyz    [B, K, Z, Y, X]
        hm_z0 = heatmaps.sum((3, 4))  # (B, K, D)

        range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device).unsqueeze(-1)  # [64,1]
        # hm_x = hm_x0 * range_tensor
        # hm_y = hm_y0 * range_tensor
        # hm_z = hm_z0 * range_tensor

        # coord_x = hm_x.sum(dim=2, keepdim=True)
        # coord_y = hm_y.sum(dim=2, keepdim=True)
        # coord_z = hm_z.sum(dim=2, keepdim=True)
        coord_x = hm_x0.matmul(range_tensor)  # hm_x0 [1, 29, 64] range_tensor [64,1]  hm_x0.sum(-1) = [1,1,1,1...]
        coord_y = hm_y0.matmul(range_tensor)  # coord_y [1, 29, 1]  32.0377, 32.0233
        coord_z = hm_z0.matmul(range_tensor)

        # 采样
        scores = []
        img_feat_joints = []
        for j in range(self.joint_num ):
            x = coord_x[:,j,0] / (self.width_dim - 1) * 2 - 1  # TODO *2-1是什么意思，前面的部分相当于该点在图片的x轴百分比 0.2892
            y = coord_y[:,j,0] / (self.height_dim - 1) * 2 - 1  # 0.2934  # 
            z = coord_z[:,j,0] / (self.depth_dim - 1) * 2 - 1  # 0.1429  [1,29,1] 之间-1~1
            # 得到关节点置信度
            grid = torch.stack((x, y, z), 1)[:, None, None, None, :]  # torch.stack((x, y, z), 1) [1,3] grid [1,1,1,1,3]
            score_j = F.grid_sample(heatmaps[:, j, None, :, :, :], grid, align_corners=True)[:, 0, 0, 0, 0]  # score_j [1(batchsize)] 因为batch_size=1,(batch_size) oint_heatmap[:, j, None, :, :, :] [N,1,64,64,64] -> [N,1,1,1,1]
            scores.append(score_j)  # TODO grid_sample

            # 根据关节点在图像特征进行采样
            img_feat = x0.float()
            img_grid = torch.stack((x, y), 1)[:, None, None, :] # [N,1,1,2] <- [N,2]
            img_feat_j = F.grid_sample(img_feat, img_grid, align_corners=True)[:, :, 0, 0]  # (batch_size, channel_dim) [1,2048]
            img_feat_joints.append(img_feat_j)
        scores = torch.stack(scores)  # (joint_num, batch_size)  [29,1]  [tensor,tensor...]15个tensor, stack默认0所以是[15,1]如果是1则是[1,15]
        joint_score = scores.permute(1, 0)[:, :, None]  # (batch_size, joint_num, 1)  [1,29,1]
        img_feat_joints = torch.stack(img_feat_joints) # (joint_num, batch_size, channel_dim) [15,1,2048]
        img_feat_joints = img_feat_joints.permute(1, 0 ,2) # (batch_size, joint_num, channel_dim) [1,15,2048]
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)  # 0, 64
        feat = torch.cat((img_feat_joints, pred_uvd_jts_29, joint_score), dim=2)  # [1,15,2052(C'+3+1=2048+3+1=2052)]
        feat = feat.view(x0.size(0), -1)  # [1, 59508]
        feat = self.dejoint(feat)  # 【1，2048】<- [1, 59508] conv(59508,2048)


        #  -0.5 ~ 0.5
        # coord_x = coord_x / float(self.width_dim) - 0.5  # coord_z坐标是像素,热力图是64像素，现在转换成以中心点为中心的百分比
        # coord_y = coord_y / float(self.height_dim) - 0.5  # 小数比例, 0-1
        # coord_z = coord_z / float(self.depth_dim) - 0.5  # coord_x [0,64] -> [-0.5,0.5]
        # pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)  # [1, 29, 3]
        pred_uvd_jts_29 = pred_uvd_jts_29/64 - 0.5

        # mesh部分
        # feat = feat.view(x0.size(0), -1)  # [1, 2048]
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)  <- [10]
        init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)

        # xc = x0

        delta_shape = self.decshape(feat)  # [1, 10] <- [1,2048]
        pred_shape = delta_shape + init_shape  # beta
        pred_phi = self.decphi(feat)  # [1,46] <- conv[2048,46] [1,2048]
        pred_camera = self.deccam(feat).reshape(batch_size, -1) + init_cam  # [1,1]
        sigma = self.decsigma(feat).reshape(batch_size, 29, 1).sigmoid()  # [1, 29,1]

        pred_phi = pred_phi.reshape(batch_size, 23, 2)  # [1, 23, 2]

        if flip_test:

            flip_delta_shape = self.decshape(flip_x0)
            flip_pred_shape = flip_delta_shape + init_shape
            flip_pred_phi = self.decphi(flip_x0)
            flip_pred_camera = self.deccam(flip_x0).reshape(batch_size, -1) + init_cam  # 相机s
            flip_sigma = self.decsigma(flip_x0).reshape(batch_size, 29, 1).sigmoid()

            pred_shape = (pred_shape + flip_pred_shape) / 2

            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2

            pred_camera = 2 / (1 / flip_pred_camera + 1 / pred_camera)

            flip_sigma = self.flip_sigma(flip_sigma)
            sigma = (sigma + flip_sigma) / 2

        camScale = pred_camera[:, :1].unsqueeze(1)  # [1,1,1] 0.4902
        # camTrans = pred_camera[:, 1:].unsqueeze(1)

        camDepth = self.focal_length / (self.input_size * camScale + 1e-9)  # 7.9685

        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']

            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5  # 471.0548
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5  # 134.1151
            w = (bboxes[:, 2] - bboxes[:, 0])  # 51.5535
            h = (bboxes[:, 3] - bboxes[:, 1])  # 51.5535  # w,h是bbox长宽

            cx = cx - img_center[:, 0]  # cx=221.0548 <- img_center 471.0548【[250.0000, 187.5000]】
            cy = cy - img_center[:, 1]  # -53.3848
            cx = cx / w  # 4.2879
            cy = cy / h  # -1.0355

            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)  # bbox_center是bbox_center到原始图像img_center的距离除上bbox_h

            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xy_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m  xy

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)  z

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        # camTrans = camera_root.squeeze(dim=1)[:, :2]

        # if not self.training:
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]  # 相机坐标变成根节点坐标

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor,  # unit: meter
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )  # pred_xyz_jts_29 [1,29,3]根节点坐标系，单位m， pred_shape betas[1,10] phis [1,23,2]
        pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 9)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        transl = camera_root - output.joints.float().reshape(-1, 24, 3)[:, 0, :]

        output = edict(
            pred_phi=pred_phi,
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1),
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=maxvals,
            cam_scale=camScale[:, 0],
            # cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            pred_camera=pred_camera,
            pred_sigma=sigma,
            scores=1 - sigma,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            img_feat=x0
        )
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):

        output = self.smpl(
            pose_axis_angle=gt_theta,
            betas=gt_beta,
            global_orient=None,
            return_verts=True
        )

        return output
