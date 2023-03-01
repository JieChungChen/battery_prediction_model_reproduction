import torch
import torch.nn as nn
from itertools import repeat


class SpatialDropout(nn.Module):
    """
    spatial dropout是針對channel位置做dropout
    ex.若對(batch, timesteps, embedding)的輸入沿着axis=1執行
    可對embedding的數個channel整體dropout
    沿着axis=2則是對某些timestep整體dropout
    """
    def __init__(self, drop=0.2):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 應和inputs的shape一致, 其中值為1的即沿著drop的axis
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1])   # 默认沿着中间所有的shape
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


def conv_block(in_ch, out_ch, kernel_size, padding, activation=True):
    if activation:
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                            nn.Mish())
    else:
        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size, padding),)


class Dim_Reduction_1(nn.Module):  # for EOL
    def __init__(self, in_ch, out_ch):
        super(Dim_Reduction_1, self).__init__()
        self.conv1_1 = conv_block(in_ch, 128, kernel_size=13, padding=6)
        self.conv1_2 = conv_block(128, 128, kernel_size=13, padding=6)
        self.conv1_3 = conv_block(128, 384, kernel_size=5, padding=2)
        self.conv1_4 = conv_block(384, 384, kernel_size=5, padding=2)
        self.maxpool1_1 = nn.MaxPool1d(2, 2)
        self.maxpool1_2 = nn.MaxPool1d(2, 2)
        self.maxpool1_3 = nn.MaxPool1d(2, 2)
        self.spacial_drop1 = SpatialDropout(0.2)
        self.conv2_1 = conv_block(512, 64, kernel_size=17, padding=8)
        self.conv2_2 = conv_block(64, 64, kernel_size=17, padding=8)
        self.conv2_3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.conv2_4 = conv_block(192, 192, kernel_size=3, padding=1)
        self.maxpool2_1 = nn.AvgPool1d(2, 2)
        self.maxpool2_2 = nn.AvgPool1d(2, 2)
        self.avgpool1 = nn.MaxPool1d(2, 2)
        self.conv3_1 = conv_block(256, 32, kernel_size=13, padding=6)
        self.conv3_2 = conv_block(32, 32, kernel_size=13, padding=6)
        self.conv3_3 = conv_block(32, 32, kernel_size=3, padding=1)
        self.conv3_4 = conv_block(32, 32, kernel_size=3, padding=1)
        self.spatial_drop2 = SpatialDropout(0.2)
        self.gloavgpool = nn.AdaptiveAvgPool1d(1)
        self.glomaxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(64, out_ch)

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_1 = self.conv1_2(conv1_1)
        conv1_2 = self.conv1_3(conv1_1)
        conv1_2 = self.conv1_4(conv1_2)
        conv1_out = self.maxpool1_3(torch.cat((self.maxpool1_1(conv1_1), self.maxpool1_2(conv1_2)), dim=1))
        conv1_out = self.spacial_drop1(conv1_out)
        conv2_1 = self.conv2_1(conv1_out)
        conv2_1 = self.conv2_2(conv2_1)
        conv2_2 = self.conv2_3(conv2_1)
        conv2_2 = self.conv2_4(conv2_2)
        conv2_out = self.avgpool1(torch.cat((self.maxpool2_1(conv2_1), self.maxpool2_2(conv2_2)), dim=1))
        conv3_1 = self.conv3_1(conv2_out)
        conv3_1 = self.conv3_2(conv3_1)
        conv3_2 = self.conv3_3(conv3_1)
        conv3_2 = self.conv3_4(conv3_2)
        conv3_out = self.spatial_drop2(torch.cat((conv3_1, conv3_2), dim=1))
        conv3_out = torch.add(self.gloavgpool(conv3_out), self.glomaxpool(conv3_out))
        out = self.linear(torch.squeeze(conv3_out))
        return out


class Dim_Reduction_2(nn.Module):  # for chargetime
    def __init__(self, in_ch, out_ch):
        super(Dim_Reduction_2, self).__init__()
        self.conv1_1 = conv_block(in_ch, 128, kernel_size=13, padding=6)
        self.conv1_2 = conv_block(128, 128, kernel_size=13, padding=6)
        self.conv1_3 = conv_block(128, 256, kernel_size=5, padding=2)
        self.conv1_4 = conv_block(256, 256, kernel_size=5, padding=2)
        self.maxpool1_1 = nn.MaxPool1d(2, 2)
        self.maxpool1_2 = nn.MaxPool1d(2, 2)
        self.maxpool1_3 = nn.AvgPool1d(2, 2)
        self.spacial_drop1 = SpatialDropout(0.2)
        self.conv2_1 = conv_block(384, 32, kernel_size=17, padding=8)
        self.conv2_2 = conv_block(32, 32, kernel_size=17, padding=8)
        self.conv2_3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.conv2_4 = conv_block(64, 64, kernel_size=3, padding=1)
        self.maxpool2_1 = nn.MaxPool1d(2, 2)
        self.maxpool2_2 = nn.MaxPool1d(2, 2)
        self.avgpool1 = nn.MaxPool1d(2, 2)
        self.conv3_1 = conv_block(96, 32, kernel_size=3, padding=1)
        self.conv3_2 = conv_block(32, 32, kernel_size=3, padding=1)
        self.conv3_3 = conv_block(32, 96, kernel_size=9, padding=4)
        self.conv3_4 = conv_block(96, 96, kernel_size=9, padding=4)
        self.spatial_drop2 = SpatialDropout(0.2)
        self.gloavgpool = nn.AdaptiveAvgPool1d(1)
        self.glomaxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(128, out_ch)

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_1 = self.conv1_2(conv1_1)
        conv1_2 = self.conv1_3(conv1_1)
        conv1_2 = self.conv1_4(conv1_2)
        conv1_out = self.maxpool1_3(torch.cat((self.maxpool1_1(conv1_1), self.maxpool1_2(conv1_2)), dim=1))
        conv1_out = self.spacial_drop1(conv1_out)
        conv2_1 = self.conv2_1(conv1_out)
        conv2_1 = self.conv2_2(conv2_1)
        conv2_2 = self.conv2_3(conv2_1)
        conv2_2 = self.conv2_4(conv2_2)
        conv2_out = self.avgpool1(torch.cat((self.maxpool2_1(conv2_1), self.maxpool2_2(conv2_2)), dim=1))
        conv3_1 = self.conv3_1(conv2_out)
        conv3_1 = self.conv3_2(conv3_1)
        conv3_2 = self.conv3_3(conv3_1)
        conv3_2 = self.conv3_4(conv3_2)
        conv3_out = self.spatial_drop2(torch.cat((conv3_1, conv3_2), dim=1))
        conv3_out = torch.add(self.gloavgpool(conv3_out), self.glomaxpool(conv3_out))
        out = self.linear(torch.squeeze(conv3_out))
        return out


class Dim_Reduction_3(nn.Module):  # for EOL
    def __init__(self, in_ch, out_ch):
        super(Dim_Reduction_3, self).__init__()
        self.conv1_1 = conv_block(in_ch, 128, kernel_size=17, padding=8)
        self.conv1_2 = conv_block(128, 128, kernel_size=17, padding=8)
        self.conv1_3 = conv_block(128, 384, kernel_size=9, padding=4)
        self.conv1_4 = conv_block(384, 384, kernel_size=9, padding=4)
        self.maxpool1_1 = nn.MaxPool1d(2, 2)
        self.maxpool1_2 = nn.MaxPool1d(2, 2)
        self.maxpool1_3 = nn.MaxPool1d(2, 2)
        self.spacial_drop1 = SpatialDropout(0.2)
        self.conv2_1 = conv_block(512, 64, kernel_size=17, padding=8)
        self.conv2_2 = conv_block(64, 64, kernel_size=17, padding=8)
        self.conv2_3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.conv2_4 = conv_block(192, 192, kernel_size=3, padding=1)
        self.maxpool2_1 = nn.AvgPool1d(2, 2)
        self.maxpool2_2 = nn.AvgPool1d(2, 2)
        self.avgpool1 = nn.MaxPool1d(2, 2)
        self.conv3_1 = conv_block(256, 64, kernel_size=13, padding=6)
        self.conv3_2 = conv_block(64, 64, kernel_size=13, padding=6)
        self.conv3_3 = conv_block(64, 64, kernel_size=3, padding=1)
        self.conv3_4 = conv_block(64, 64, kernel_size=3, padding=1)
        self.spatial_drop2 = SpatialDropout(0.2)
        self.gloavgpool = nn.AdaptiveAvgPool1d(1)
        self.glomaxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(128, out_ch)

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_1 = self.conv1_2(conv1_1)
        conv1_2 = self.conv1_3(conv1_1)
        conv1_2 = self.conv1_4(conv1_2)
        conv1_out = self.maxpool1_3(torch.cat((self.maxpool1_1(conv1_1), self.maxpool1_2(conv1_2)), dim=1))
        conv1_out = self.spacial_drop1(conv1_out)
        conv2_1 = self.conv2_1(conv1_out)
        conv2_1 = self.conv2_2(conv2_1)
        conv2_2 = self.conv2_3(conv2_1)
        conv2_2 = self.conv2_4(conv2_2)
        conv2_out = self.avgpool1(torch.cat((self.maxpool2_1(conv2_1), self.maxpool2_2(conv2_2)), dim=1))
        conv3_1 = self.conv3_1(conv2_out)
        conv3_1 = self.conv3_2(conv3_1)
        conv3_2 = self.conv3_3(conv3_1)
        conv3_2 = self.conv3_4(conv3_2)
        conv3_out = self.spatial_drop2(torch.cat((conv3_1, conv3_2), dim=1))
        conv3_out = torch.add(self.gloavgpool(conv3_out), self.glomaxpool(conv3_out))
        out = self.linear(torch.squeeze(conv3_out))
        return out


class Dim_Reduction_4(nn.Module):  # for chargetime
    def __init__(self, in_ch, out_ch):
        super(Dim_Reduction_4, self).__init__()
        self.conv1_1 = conv_block(in_ch, 256, kernel_size=15, padding=7)
        self.conv1_2 = conv_block(256, 256, kernel_size=15, padding=7)
        self.conv1_3 = conv_block(256, 512, kernel_size=3, padding=1)
        self.conv1_4 = conv_block(512, 512, kernel_size=3, padding=1)
        self.maxpool1_1 = nn.AvgPool1d(2, 2)
        self.maxpool1_2 = nn.AvgPool1d(2, 2)
        self.maxpool1_3 = nn.AvgPool1d(2, 2)
        self.spacial_drop1 = SpatialDropout(0.2)
        self.conv2_1 = conv_block(768, 32, kernel_size=3, padding=1)
        self.conv2_2 = conv_block(32, 32, kernel_size=3, padding=1)
        self.conv2_3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.conv2_4 = conv_block(64, 64, kernel_size=3, padding=1)
        self.maxpool2_1 = nn.MaxPool1d(2, 2)
        self.maxpool2_2 = nn.MaxPool1d(2, 2)
        self.avgpool1 = nn.AvgPool1d(2, 2)
        self.conv3_1 = conv_block(96, 128, kernel_size=17, padding=8)
        self.conv3_2 = conv_block(128, 128, kernel_size=17, padding=8)
        self.conv3_3 = conv_block(128, 256, kernel_size=7, padding=3)
        self.conv3_4 = conv_block(256, 256, kernel_size=7, padding=3)
        self.spatial_drop2 = SpatialDropout(0.2)
        self.gloavgpool = nn.AdaptiveAvgPool1d(1)
        self.glomaxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(384, out_ch)

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_1 = self.conv1_2(conv1_1)
        conv1_2 = self.conv1_3(conv1_1)
        conv1_2 = self.conv1_4(conv1_2)
        conv1_out = self.maxpool1_3(torch.cat((self.maxpool1_1(conv1_1), self.maxpool1_2(conv1_2)), dim=1))
        conv1_out = self.spacial_drop1(conv1_out)
        conv2_1 = self.conv2_1(conv1_out)
        conv2_1 = self.conv2_2(conv2_1)
        conv2_2 = self.conv2_3(conv2_1)
        conv2_2 = self.conv2_4(conv2_2)
        conv2_out = self.avgpool1(torch.cat((self.maxpool2_1(conv2_1), self.maxpool2_2(conv2_2)), dim=1))
        conv3_1 = self.conv3_1(conv2_out)
        conv3_1 = self.conv3_2(conv3_1)
        conv3_2 = self.conv3_3(conv3_1)
        conv3_2 = self.conv3_4(conv3_2)
        conv3_out = self.spatial_drop2(torch.cat((conv3_1, conv3_2), dim=1))
        conv3_out = torch.add(self.gloavgpool(conv3_out), self.glomaxpool(conv3_out))
        out = self.linear(torch.squeeze(conv3_out))
        return out


class Predictor_3(nn.Module):  # for EOL and discharge time
    def __init__(self, in_ch, out_ch, drop=0.2):
        super(Predictor_3, self).__init__()
        self.conv1_1 = conv_block(in_ch, 256, kernel_size=5, padding=2)
        self.conv1_2 = conv_block(256, 128, kernel_size=9, padding=4)
        self.conv1_3 = conv_block(128, 512, kernel_size=3, padding=1)
        self.spatial_drop1 = SpatialDropout(drop)
        self.avgpool1 = nn.MaxPool1d(2, 2)
        # attention layer
        self.conv2_1 = conv_block(512, 32, kernel_size=5, padding=2)
        self.conv2_2 = conv_block(512, 32, kernel_size=9, padding=4)
        self.gloavgpool1 = nn.AdaptiveAvgPool1d(1)
        self.gloavgpool2 = nn.AdaptiveAvgPool1d(1)
        self.conv3 = nn.Sequential(
            conv_block(1, 128, kernel_size=9, padding=4),
            conv_block(128, 256, kernel_size=15, padding=7),
            conv_block(256, 64, kernel_size=5, padding=2)
        )
        self.glomaxpool3 = nn.AdaptiveMaxPool1d(1)
        self.gloavgpool3 = nn.AdaptiveAvgPool1d(1)
        self.conv4 = nn.Sequential(
            conv_block(1, 32, kernel_size=7, padding=3),
            conv_block(32, 128, kernel_size=5, padding=2),
            conv_block(128, 128, kernel_size=9, padding=4)
        )
        self.glomaxpool4 = nn.AdaptiveMaxPool1d(1)
        self.gloavgpool4 = nn.AdaptiveAvgPool1d(1)
        self.conv5 = nn.Sequential(
            conv_block(1, 256, kernel_size=13, padding=6),
            conv_block(256, 256, kernel_size=7, padding=3),
            conv_block(256, 64, kernel_size=7, padding=3)
        )
        self.glomaxpool5 = nn.AdaptiveMaxPool1d(1)
        self.gloavgpool6 = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(64, out_ch)
        self.linear2 = nn.Linear(128, out_ch)
        self.linear3 = nn.Linear(64, out_ch)
    
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.spatial_drop1(x)
        x = self.avgpool1(x)
        att1 = self.conv2_1(x)
        att2 = self.conv2_2(x)
        x = torch.matmul(torch.transpose(att1, 1, 2), att2)
        x = torch.cat((self.gloavgpool1(att1), self.gloavgpool2(x)), dim=1).squeeze().unsqueeze(1)
        conv3 = self.conv3(x)
        conv3 = torch.add(self.glomaxpool3(conv3), self.gloavgpool3(conv3)).squeeze()
        conv4 = self.conv4(x)
        conv4 = torch.add(self.glomaxpool4(conv4), self.gloavgpool4(conv4)).squeeze()
        conv5 = self.conv5(x)
        conv5 = torch.add(self.glomaxpool5(conv5), self.gloavgpool5(conv5)).squeeze()
        out_eol = self.linear1(conv3)+self.linear2(conv4)+self.linear3(conv5)
        return out_eol