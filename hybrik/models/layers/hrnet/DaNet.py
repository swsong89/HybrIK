import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=False, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
    def segp(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)
        return feat_p

    def segc(self, x):
        feat_c = self.conv_c1(x)
        feat_c = self.pam(feat_c)
        feat_c = self.conv_c2(feat_c)
        return feat_c
    

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)
            
        return outputs  # aux=True的话输出有3个，sum fusion, p_out, c_out



        # checkpoint
        # x = Variable(x,requires_grad=True) 

        # feat_p = checkpoint(self.segp, x)
        # feat_c = checkpoint(self.segc, x)


        # feat_fusion = feat_p + feat_c

        # outputs = []
        # fusion_out = checkpoint(self.out, feat_fusion)
        # outputs.append(fusion_out)
        # if self.aux:
        #     p_out = self.conv_p3(feat_p)
        #     c_out = self.conv_c3(feat_c)
        #     outputs.append(p_out)
        #     outputs.append(c_out)

        # return tuple(outputs)  # aux=True的话输出有3个，sum fusion, p_out, c_out

if __name__ == "__main__":
    # t = torch.ones((2, 32, 128, 128))
    # Da = _DAHead(32, 32)
    # out = Da(t)[0]
    # print('out.shape: {}'.format(out.shape))
    # print('out.shape: ', out.shape)


    import torch
    import inspect
    from gpu_tracker import  MemTracker
    from torch.utils.checkpoint import checkpoint_sequential

    device_id = 3
    device = torch.device('cuda:{}'.format(device_id))

    frame = inspect.currentframe()          # define a frame to track
    gpu_tracker = MemTracker(frame, device=device_id)         # define a GPU tracker

    gpu_tracker.track()
    Da = _DAHead(32, 32).to(device)  # 初始化模型1781.5Mb
    gpu_tracker.track()
    input = torch.ones((2, 32, 128, 128), requires_grad=True).to(device)  #  batchsize 2 21.0MB
    gpu_tracker.track()
    out = Da(input)  #  运算4653.6Mb 
    # output = checkpoint_sequential(Da, 2, input)
    gpu_tracker.track()  #  2batch_size总共 6456.1Mb # 8 batchsize 19G

    print('ok')



    # from torchsummary import summary
    # # import os
    # # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # device = 'cpu'

    # Da = _DAHead(32, 32)  # 初始化模型1781.5Mb

    # model = nn.Sequential(Da).to(device)
    # summary(model, (32,128,128), batch_size=8, device=device)
    # # print('out.shape: {}'.format(out.shape))
    # # print('out.shape: ', out.shape)

# https://blog.csdn.net/lucifer479/article/details/125849933


"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [8, 8, 128, 128]           2,304
       BatchNorm2d-2           [8, 8, 128, 128]              16
              ReLU-3           [8, 8, 128, 128]               0
            Conv2d-4           [8, 1, 128, 128]               9
            Conv2d-5           [8, 1, 128, 128]               9
           Softmax-6          [8, 16384, 16384]               0
            Conv2d-7           [8, 8, 128, 128]              72
_PositionAttentionModule-8           [8, 8, 128, 128]               0
            Conv2d-9           [8, 8, 128, 128]             576
      BatchNorm2d-10           [8, 8, 128, 128]              16
             ReLU-11           [8, 8, 128, 128]               0
           Conv2d-12           [8, 8, 128, 128]           2,304
      BatchNorm2d-13           [8, 8, 128, 128]              16
             ReLU-14           [8, 8, 128, 128]               0
          Softmax-15                  [8, 8, 8]               0
_ChannelAttentionModule-16           [8, 8, 128, 128]               0
           Conv2d-17           [8, 8, 128, 128]             576
      BatchNorm2d-18           [8, 8, 128, 128]              16
             ReLU-19           [8, 8, 128, 128]               0
          Dropout-20           [8, 8, 128, 128]               0
           Conv2d-21          [8, 32, 128, 128]             288
          _DAHead-22       [[-1, 32, 128, 128]]               0
================================================================
Total params: 6,202
Trainable params: 6,202
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 16.00
Forward/backward pass size (MB): 16542.00
Params size (MB): 0.02
Estimated Total Size (MB): 16558.03


"""