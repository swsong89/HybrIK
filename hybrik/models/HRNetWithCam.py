from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import SPPE
from .layers.smpl.SMPL import SMPL_layer
from .layers.hrnet.hrnet import get_hrnet
from .keypoint_attention import KeypointAttention

is_dev_sample = False
is_dev_relu = False

is_dev_sample = True
# is_dev_relu = True

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



        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer(
            'init_shape',
            torch.Tensor(init_shape).float())

        init_cam = torch.tensor([0.9])
        self.register_buffer(
            'init_cam',
            torch.Tensor(init_cam).float())


        if is_dev_sample:
            # 下面是基于关节点
            # self.joint_num = 29  # 15
            # # self.graph_adj = torch.from_numpy(self.smpl.graph_adj).float()  # 15x15

            # # # graph convs  2048为C',即F'特征2048x8x8 4=3+1为P 3D +置信度
            # # self.graph_block = nn.Sequential(*[\
            # #     GraphConvBlock(self.graph_adj, 2048+4, 128),
            # #     GraphResBlock(self.graph_adj, 128),
            # #     GraphResBlock(self.graph_adj, 128),
            # #     GraphResBlock(self.graph_adj, 128),
            # #     GraphResBlock(self.graph_adj, 128)])
            self.heatmap_conv = nn.Conv2d(
                        in_channels=48,
                        out_channels=1856,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
            if is_dev_relu:
              self.relu = nn.ReLU(True)
            self.num_joint = 29
            self.dejoint = nn.Linear(3072, 1024)
            self.decshape = nn.Linear(1024, 10)
            self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
            self.deccam = nn.Linear(1024, 1)
            self.decsigma = nn.Linear(1024, 29)
        else:
        # 下面是原始的
            self.decshape = nn.Linear(2048, 10)
            self.decphi = nn.Linear(2048, 23 * 2)  # [cos(phi), sin(phi)]
            self.deccam = nn.Linear(2048, 1)
            self.decsigma = nn.Linear(2048, 29)

        self.focal_length = kwargs['FOCAL_LENGTH']
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        self.input_size = 256.0


        num_deconv_layers=3
        num_deconv_filters=(256, 256, 256)
        num_deconv_kernels=(4, 4, 4)
        use_upsampling = False
        self.num_input_features = 48  # hrbackbone输出的特征通道维度  1856
        
        conv_fn = self._make_upsample_layer if use_upsampling else self._make_deconv_layer

        self.keypoint_deconv_layers = nn.Conv2d(
                        in_channels=self.num_input_features,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )

        self.smpl_deconv_layers = nn.Conv2d(
                        in_channels=self.num_input_features,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )

        self.keypoint_final_layer = nn.Conv2d(
                        in_channels=128,
                        out_channels=24+1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )

        self.keypoint_attention = KeypointAttention(
                        use_conv=False,
                        in_channels=(128,64),
                        out_channels=(128,64),
                        act='softmax',
                        use_scale=False,
                    )


    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    # def _make_upsample_layer(self, num_layers, num_filters, num_kernels):
    #     assert num_layers == len(num_filters), \
    #         'ERROR: num_layers is different len(num_filters)'
    #     assert num_layers == len(num_kernels), \
    #         'ERROR: num_layers is different len(num_filters)'

    #     layers = []
    #     for i in range(num_layers):
    #         kernel, padding, output_padding = \
    #             self._get_deconv_cfg(num_kernels[i])

    #         planes = num_filters[i]
    #         layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
    #         layers.append(
    #             nn.Conv2d(in_channels=self.num_input_features, out_channels=planes,
    #                     kernel_size=kernel, stride=1, padding=padding, bias=False)
    #         )
    #         layers.append(nn.BatchNorm2d(planes, momentum=0.1))
    #         layers.append(nn.ReLU(inplace=True))
    #         # if self.use_self_attention:
    #         #     layers.append(SelfAttention(planes))
    #         self.num_input_features = planes
    #     return nn.Sequential(*layers)


    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            # if self.use_self_attention:
            #     layers.append(SelfAttention(planes))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _get_2d_branch_feats(self, features):  # 11, 480, 56, 56
        part_feats = self.keypoint_deconv_layers(features)  # 11, 128, 56, 56
        # if self.use_branch_nonlocal:
        #     part_feats = self.branch_2d_nonlocal(part_feats)
        return part_feats

    def _get_part_attention_map(self, part_feats):
        heatmaps = self.keypoint_final_layer(part_feats)  # [11, 25, 56, 56] <- 11, 128, 56, 56
        heatmaps = heatmaps[:,1:,:,:] # remove the first channel which encodes the background
        return heatmaps

    def _get_3d_smpl_feats(self, features, part_feats):
        use_keypoint_features_for_smpl_regression = False
        if use_keypoint_features_for_smpl_regression:
            smpl_feats = part_feats
        else:
            smpl_feats = self.smpl_deconv_layers(features)  # 11, 128, 56, 56
        return smpl_feats

    def _get_local_feats(self, smpl_feats, part_attention):  # 11, 128, 56, 56  11, 24, 56, 56
        # cam_shape_feats = self.smpl_final_layer(smpl_feats)  # 11, 64, 56, 56

        point_local_feat = self.keypoint_attention(smpl_feats, part_attention)  # [N.C.js] 11, 128, 24  <- 11, 128, 56, 56  11, 24, 56, 56
        # cam_shape_feats = self.keypoint_attention(cam_shape_feats, part_attention)  # [N.C.js] 11, 64, 24 <- 11, 64, 56, 56  11, 24, 56, 56
        
        return point_local_feat  # 11, 128, 24   11, 64, 24



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
        out = self.preact(x)  # [1, 1856, 64, 64] [1, 2048]
        # print(out.shape)
        heat_out = self.heatmap_conv(out)
        heat_out = heat_out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)  # [1, 29, 64, 64, 64]

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
            heat_out = heat_out.reshape((out.shape[0], self.num_joints, -1))

            heatmaps = norm_heatmap(self.norm_type, heat_out)

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
        if is_dev_sample:
            if flip_test == False:
                # 只是简单的将基于关键点的特征提取换成pare部分，具体代码还得修改
                out = out.reshape((out.shape[0], -1, 64, 64))
                part_feats = self._get_2d_branch_feats(out)  #  11, 128, 56, 56 <- features 11, 480, 56, 56

                ############## GET PART ATTENTION MAP ##############
                part_attention = self._get_part_attention_map(part_feats)  # 11, 24, 56, 56  <- 11, 128, 56, 56 , 25第一个是mask,所以24

                ############## 3D SMPL BRANCH FEATURES ##############
                smpl_feats = self._get_3d_smpl_feats(out, part_feats)  # 使用out进行翻卷积或者使用part_feat来相乘，后这就是相同特征进行attetion且相乘
                point_local_feat = self._get_local_feats(smpl_feats, part_attention)  # [16384, 24] 11, 128, 24   11, 64, 24  <- 11, 128, 56, 56  11, 24, 56, 56
                # feat = torch.cat((img_feat_joints, pred_uvd_jts_29, joint_score), dim=2)  # [1,15,2052(C'+3+1=2048+3+1=2052)]
                feat = point_local_feat.view(out.size(0), -1)  # [1, 59508]
                feat = self.dejoint(feat)  # 【1，2048】<- [1, 59508] conv(59508,2048)
                # print('before feat min: ', feat.detach().min().cpu().numpy(), ' feat max: ', feat.detach().max().cpu().numpy())
                # feat = torch.sigmoid(feat)
                # 和softmax替换
                if is_dev_relu:
                    feat = self.relu(feat)
                    feat = (torch.sigmoid(feat)-0.5)*2
                    # feat = self.relu()
                    # torch.tanh(feat, feat)
                else:
                    feat = (feat-feat.min(-1).values.view(-1,1))/(feat.max(-1).values.view(-1,1)-feat.min(-1).values.view(-1,1))
                pred_uvd_jts_29 = torch.cat([coord_x, coord_y, coord_z], -1)
                pred_uvd_jts_29 = pred_uvd_jts_29/64 - 0.5

                init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)  <- [10]
                init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)
                delta_shape = self.decshape(feat)  # [1, 10] <- [1,2048]
                pred_shape = delta_shape + init_shape  # beta
                pred_phi = self.decphi(feat)  # [1,46] <- conv[2048,46] [1,2048]
                pred_camera = self.deccam(feat).reshape(batch_size, -1) + init_cam  # [1,1]
                sigma = self.decsigma(feat).reshape(batch_size, 29, 1).sigmoid()  # [1, 29,1]

                pred_phi = pred_phi.reshape(batch_size, 23, 2)  # [1, 23, 2]
                # print('dev feat min: ', feat.detach().min().cpu().numpy(), ' feat max: ', feat.detach().max().cpu().numpy(), \
                #         ' camera_scale: ', pred_camera.detach().cpu().numpy()[0,0])
            else:  # dev flip_test
                out = out.reshape((out.shape[0], -1, 64, 64))
                part_feats = self._get_2d_branch_feats(out)  #  11, 128, 56, 56 <- features 11, 480, 56, 56

                ############## GET PART ATTENTION MAP ##############
                part_attention = self._get_part_attention_map(part_feats)  # 11, 24, 56, 56  <- 11, 128, 56, 56 , 25第一个是mask,所以24

                ############## 3D SMPL BRANCH FEATURES ##############
                smpl_feats = self._get_3d_smpl_feats(out, part_feats)  # 使用out进行翻卷积或者使用part_feat来相乘，后这就是相同特征进行attetion且相乘
                point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention)  # [16384, 24] 11, 128, 24   11, 64, 24  <- 11, 128, 56, 56  11, 24, 56, 56
                # feat = torch.cat((img_feat_joints, pred_uvd_jts_29, joint_score), dim=2)  # [1,15,2052(C'+3+1=2048+3+1=2052)]
                feat = point_local_feat.view(out.size(0), -1)  # [1, 59508]
                feat = self.dejoint(feat)  # 【1，2048】<- [1, 59508] conv(59508,2048)
                if is_dev_relu:
                    feat = self.relu(feat)
                    feat = (torch.sigmoid(feat)-0.5)*2
                else:
                    feat = (feat-feat.min(-1).values.view(-1,1))/(feat.max(-1).values.view(-1,1)-feat.min(-1).values.view(-1,1))                # feat = torch.sigmoid(feat)
                # print('before feat min: ', feat.detach().min().cpu().numpy(), ' feat max: ', feat.detach().max().cpu().numpy())

                img_flip_feat_joints = torch.stack(img_flip_feat_joints) # (joint_num, batch_size, channel_dim) [15,1,2048]
                img_flip_feat_joints = img_flip_feat_joints.permute(1, 0 ,2) # (batch_size, joint_num, channel_dim) [1,15,2048]
                flip_feat = torch.cat((img_flip_feat_joints, pred_uvd_jts_29, joint_score), dim=2)  # [1,15,2052(C'+3+1=2048+3+1=2052)]
                flip_feat = flip_feat.view(x0.size(0), -1)  # [1, 59508]
                flip_feat = self.dejoint(flip_feat)  # 【1，2048】<- [1, 59508] conv(59508,2048)
                if is_dev_relu:
                    feat = self.relu(feat)
                    feat = (torch.sigmoid(feat)-0.5)*2
                else:
                    flip_feat = (flip_feat-flip_feat.min(-1).values.view(-1,1))/(flip_feat.max(-1).values.view(-1,1)-flip_feat.min(-1).values.view(-1,1))                # feat = torch.sigmoid(feat)
                # 和softmax替换

                pred_uvd_jts_29 = pred_uvd_jts_29/64 - 0.5

                # feat = feat.view(x0.size(0), -1)  # [1, 2048]
                init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)  <- [10]
                init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)

                # 正常预测
                delta_shape = self.decshape(feat)  # [1, 10] <- [1,2048]
                pred_shape = delta_shape + init_shape  # beta
                pred_phi = self.decphi(feat).reshape(batch_size, 23, 2)  # [1,46] <- conv[2048,46] [1,2048]
                pred_camera = self.deccam(feat).reshape(batch_size, -1) + init_cam  # [1,1]
                sigma = self.decsigma(feat).reshape(batch_size, 29, 1).sigmoid()  # [1, 29,1]

                # print('feat min: ', feat.detach().min().cpu().numpy(), ' feat max: ', feat.detach().max().cpu().numpy(), \
                #         ' camera_scale: ', pred_camera.detach().cpu().numpy()[0,0])

                # flip预测
                flip_delta_shape = self.decshape(flip_feat)
                flip_pred_shape = flip_delta_shape + init_shape
                flip_pred_phi = self.decphi(flip_feat).reshape(batch_size, 23, 2)
                flip_pred_camera = self.deccam(flip_feat).reshape(batch_size, -1) + init_cam  # 相机s
                flip_sigma = self.decsigma(flip_feat).reshape(batch_size, 29, 1).sigmoid()

                pred_shape = (pred_shape + flip_pred_shape) / 2

                flip_pred_phi = self.flip_phi(flip_pred_phi)
                pred_phi = (pred_phi + flip_pred_phi) / 2

                pred_camera = 2 / (1 / flip_pred_camera + 1 / pred_camera)

                flip_sigma = self.flip_sigma(flip_sigma)
                sigma = (sigma + flip_sigma) / 2
        else:
            # 正常
            #  -0.5 ~ 0.5
            feat = F.avg_pool2d(x0, kernel_size=x0.size()
                                            [2:]).view(x0.size(0), -1)
            # feat = x0
            coord_x = coord_x / float(self.width_dim) - 0.5  # coord_z坐标是像素,热力图是64像素，现在转换成以中心点为中心的百分比
            coord_y = coord_y / float(self.height_dim) - 0.5  # 小数比例, 0-1
            coord_z = coord_z / float(self.depth_dim) - 0.5  # coord_x [0,64] -> [-0.5,0.5]
            pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)  # [1, 29, 3]

            # feat = feat.view(x0.size(0), -1)  # [1, 2048]
            init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)  <- [10]
            init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)

            delta_shape = self.decshape(feat)  # [1, 10] <- [1,2048]
            pred_shape = delta_shape + init_shape  # beta
            pred_phi = self.decphi(feat)  # [1,46] <- conv[2048,46] [1,2048]
            pred_camera = self.deccam(feat).reshape(batch_size, -1) + init_cam  # [1,1]
            sigma = self.decsigma(feat).reshape(batch_size, 29, 1).sigmoid()  # [1, 29,1]

            pred_phi = pred_phi.reshape(batch_size, 23, 2)  # [1, 23, 2]
            # print('normal feat min: ', feat.detach().min().cpu().numpy(), ' feat max: ', feat.detach().max().cpu().numpy(), \
            #         ' camera_scale: ', pred_camera.detach().cpu().numpy()[0,0])

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
            img_feat=out
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
