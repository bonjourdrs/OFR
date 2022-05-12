import torch
import torch.nn as nn
from time import time
from model.quant_dorefa import *

W_BIT = 8
conv2d_q = conv2d_Q_fn(W_BIT)
linear_q = linear_Q_fn(W_BIT)
# batchNorm2d_q = batchNorm2d_Q_fn(W_BIT)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ConvBn(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            conv2d_q(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
            # batchNorm2d_q(out_c)


        )

    def forward(self, x):
        return self.net(x)


class ConvBnPrelu(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride, padding, groups),
            activation_quantize_fn(W_BIT)
        )

    def forward(self, x):
        return self.net(x)


class DepthWise(nn.Module):

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu(in_c, groups, kernel=(1, 1), stride=1, padding=0),
            ConvBnPrelu(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups),
            ConvBn(groups, out_c, kernel=(1, 1), stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)


class DepthWiseRes(nn.Module):
    """DepthWise with Residual"""

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = DepthWise(in_c, out_c, kernel, stride, padding, groups)

    def forward(self, x):
        return self.net(x) + x


class MultiDepthWiseRes(nn.Module):

    def __init__(self, num_block, channels, kernel=(3, 3), stride=1, padding=1, groups=1):
        super().__init__()

        self.net = nn.Sequential(*[
            DepthWiseRes(channels, channels, kernel, stride, padding, groups)
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.net(x)


class PyramidFeatures_qua(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures_qua, self).__init__()
        # upsample C5 to get P5 from the FPN paper

        self.P5_1 = conv2d_q(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P5 elementwise to C4
        self.P4_1 = conv2d_q(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P4 elementwise to C3
        self.P3_1 = conv2d_q(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = conv2d_q(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return P3_x

class MobileNet_Occ_qua(nn.Module):

    def __init__(self, filter_list, is_gray):
        super().__init__()
        if is_gray:
            self.conv1 = ConvBnPrelu(1, filter_list[0], kernel=(3, 3), stride=2, padding=1)
        else:
            self.conv1 = ConvBnPrelu(3, filter_list[0], kernel=(3, 3), stride=2, padding=1)
        self.conv2 = ConvBn(filter_list[0], filter_list[1], kernel=(3, 3), stride=1, padding=1, groups=filter_list[0])

        self.conv3 = DepthWise(filter_list[1], filter_list[2], kernel=(3, 3), stride=2, padding=1, groups=2 * filter_list[2])
        self.conv4 = MultiDepthWiseRes(num_block=4, channels=filter_list[2], kernel=3, stride=1, padding=1, groups=2 * filter_list[2])

        self.conv5 = DepthWise(filter_list[2], filter_list[3], kernel=(3, 3), stride=2, padding=1, groups=2 * filter_list[3])
        self.conv6 = MultiDepthWiseRes(num_block=6, channels=filter_list[3], kernel=(3, 3), stride=1, padding=1, groups=2 * filter_list[3])

        self.conv7 = DepthWise(filter_list[3], filter_list[4], kernel=(3, 3), stride=2, padding=1, groups=2 * filter_list[4])
        self.conv8 = MultiDepthWiseRes(num_block=2, channels=filter_list[4], kernel=(3, 3), stride=1, padding=1, groups=2 * filter_list[4])

        # --Begin--Triplet branch
        self.mask = nn.Sequential(
            conv2d_q(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            # batchNorm2d_q(256),
            nn.BatchNorm2d(256),
            conv2d_q(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.fpn = PyramidFeatures_qua(filter_list[2], filter_list[3], filter_list[4])

        self.regress = nn.Sequential(
            nn.BatchNorm1d(filter_list[4]*7*6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            linear_q(filter_list[4]*7*6, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )
        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            linear_q(filter_list[4] * 7 * 6, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self, x, mask=None):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        x2 = self.conv4(out)
        out = self.conv5(x2)
        x3 = self.conv6(out)
        out = self.conv7(x3)
        fmap = self.conv8(out)

        # generate mask
        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            mask = self.mask(features)

        # regress
        vec = self.regress(mask.view(mask.size(0), -1))

        fmap_mask = fmap * mask

        fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))

        fc = self.fc(fmap.view(fmap.size(0), -1))

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

def FaceMobileNet_Occ_qua(num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    is_gray = False
    # filter_list = [32, 32, 64, 128, 256, num_mask]
    model = MobileNet_Occ_qua(filter_list, is_gray)
    return model

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    x = Image.new('RGB', [96,112], (128,128,128)).convert('L')
    x = x.resize((112, 96))
    x = np.asarray(x, dtype=np.float32)
    x = x[None, None, ...]
    x = torch.from_numpy(x)

    net = FaceMobileNet_Occ()
    net.eval()
    with torch.no_grad():
        out = net(x)
    print(out.shape)