from torch import nn
from torch.utils.checkpoint import checkpoint


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):  # 通道减少16倍reduction
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # [8, 32, 128, 128]
        y = self.avg_pool(x).view(b, c)  # [8, 32]
        y = self.fc(y).view(b, c, 1, 1)  # [8, 32, 1, 1]
        return x * y.expand_as(x)  # x [8, 32, 128, 128] y [8, 32, 128, 128]

        # b, c, _, _ = x.size()  # [8, 32, 128, 128]
        # y = self.avg_pool(x).view(b, c)  # [8, 32]
        # # y = checkpoint(self.avg_pool, x).view(b, c) # [8, 32]
        # y = checkpoint(self.fc, y).view(b, c, 1, 1)  # [8, 32, 1, 1]
        # return x * y.expand_as(x)  # x [8, 32, 128, 128] y [8, 32, 128, 128]<- [8,32,1,1]


if __name__ == '__main__':

    import torch
    import inspect

    from gpu_tracker import  MemTracker
    device_id = 0
    device = torch.device('cuda:{}'.format(device_id))

    frame = inspect.currentframe()          # define a frame to track
    gpu_tracker = MemTracker(frame, device=device_id)         # define a GPU tracker
    gpu_tracker.track()
    se = SELayer(channel=32, reduction=8)  # channel=32, reduction=4 1781MB  channel=32, reduction=8 1781.MB
    gpu_tracker.track()
    # input = torch.ones((8, 32, 128, 128)).to(device)  #  batchsize 2 21.0MB
    input = torch.ones((8, 32, 128, 128), requires_grad=True).to(device)
    gpu_tracker.track()
    out = se.to(device)(input)  # 333.4Mb 
    gpu_tracker.track()  # 总共 2138.8 Mb  # 2batch_size 2G 8batch_size 7G

    # print('ok')







    # from torchsummary import summary
    # # import os
    # # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # device = 'cpu'

    # se = SELayer(channel=32, reduction=8)  # channel=32, reduction=4 1781MB  channel=32, reduction=8 1781.MB
    # model = nn.Sequential(se).to(device)
    # summary(model, (32,128,128), batch_size=8, device=device)
    # # print('out.shape: {}'.format(out.shape))
    # # print('out.shape: ', out.shape)

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
 AdaptiveAvgPool2d-1              [8, 32, 1, 1]               0
            Linear-2                     [8, 4]             128
              ReLU-3                     [8, 4]               0
            Linear-4                    [8, 32]             128
           Sigmoid-5                    [8, 32]               0
           SELayer-6          [8, 32, 128, 128]               0
================================================================
Total params: 256
Trainable params: 256
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 16.00
Forward/backward pass size (MB): 32.01
Params size (MB): 0.00
Estimated Total Size (MB): 48.01


"""