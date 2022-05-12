import torch
import torch.nn as nn
from time import time

class PyramidFeatures2(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures2, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        # self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

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

        # P6_x = self.P6(C5)

        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)

        return P3_x

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class LResNet_Occ(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_Occ, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        # --Begin--Triplet branch 
        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
            # nn.PReLU(filter_list[4]),
            # nn.BatchNorm2d(filter_list[4]),
            nn.Sigmoid(),
        ) 
        self.fpn = PyramidFeatures(filter_list[2], filter_list[3], filter_list[4])

        self.regress = nn.Sequential(
            nn.BatchNorm1d(filter_list[4]*7*6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4]*7*6, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )
        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 7 * 6, 512),
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


    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):

        t0 =time()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        fmap = self.layer4(x3)

        t1 =time()

        # generate mask
        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            mask = self.mask(features[0])

        t2 =time()
        # regress
        vec = self.regress(mask.view(mask.size(0), -1))

        t3 =time()
        fmap_mask = fmap * mask

        fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))

        t4 =time()
        fc = self.fc(fmap.view(fmap.size(0), -1))
        # print( f"trunk time: {t1-t0:.5f} \n"
        #        f"mask_fpn time: {t2 - t1:.5f} \n"
        #        f"mask_fc time: {t3 - t2:.5f} \n"
        #        f"FR_fc time: {t4 - t3:.5f} \n"
        #        )
        return fc_mask, mask, vec, fc 

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ(num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    is_gray = False
    model = LResNet_Occ(BlockIR, layers, filter_list, is_gray)
    return model 


class ConvBn(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        return self.net(x)


class ConvBnPrelu(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride, padding, groups),
            nn.PReLU(out_c)
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



class MobileNet_Occ(nn.Module):

    def __init__(self, filter_list, is_gray):
        super().__init__()
        if is_gray:
            self.conv1 = ConvBnPrelu(1, filter_list[0], kernel=(3, 3), stride=2, padding=1)
        else:
            self.conv1 = ConvBnPrelu(3, filter_list[0], kernel=(3, 3), stride=2, padding=1)
        # self.conv1 = ConvBnPrelu(1, 64, kernel=(3, 3), stride=2, padding=1)
        self.conv2 = ConvBn(filter_list[0], filter_list[1], kernel=(3, 3), stride=1, padding=1, groups=filter_list[0])

        self.conv3 = DepthWise(filter_list[1], filter_list[2], kernel=(3, 3), stride=2, padding=1, groups=2 * filter_list[2])
        self.conv4 = MultiDepthWiseRes(num_block=4, channels=filter_list[2], kernel=3, stride=1, padding=1, groups=2 * filter_list[2])

        self.conv5 = DepthWise(filter_list[2], filter_list[3], kernel=(3, 3), stride=2, padding=1, groups=2 * filter_list[3])
        self.conv6 = MultiDepthWiseRes(num_block=6, channels=filter_list[3], kernel=(3, 3), stride=1, padding=1, groups=2 * filter_list[3])

        self.conv7 = DepthWise(filter_list[3], filter_list[4], kernel=(3, 3), stride=2, padding=1, groups=2 * filter_list[4])
        self.conv8 = MultiDepthWiseRes(num_block=2, channels=filter_list[4], kernel=(3, 3), stride=1, padding=1, groups=2 * filter_list[4])

        # --Begin--Triplet branch
        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
            # nn.PReLU(filter_list[4]),
            # nn.BatchNorm2d(filter_list[4]),
            nn.Sigmoid(),
        )
        self.fpn = PyramidFeatures(filter_list[2], filter_list[3], filter_list[4])

        self.regress = nn.Sequential(
            nn.BatchNorm1d(filter_list[4]*7*6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4]*7*6, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )
        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 7 * 6, 512),
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
            mask = self.mask(features[0])

        # regress
        vec = self.regress(mask.view(mask.size(0), -1))

        fmap_mask = fmap * mask

        fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))

        fc = self.fc(fmap.view(fmap.size(0), -1))

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

def FaceMobileNet_Occ(num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    is_gray = False
    # filter_list = [32, 32, 64, 128, 256, num_mask]
    model = MobileNet_Occ(filter_list, is_gray)
    return model

class MobileNet_Occ_L(nn.Module):

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
        self.conv9 = ConvBnPrelu(filter_list[4], filter_list[4], kernel=(3, 3), stride=2, padding=1)
        self.conv10 = ConvBnPrelu(filter_list[4], filter_list[4], kernel=(3, 3), stride=2, padding=1)
        # --Begin--Triplet branch
        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.fpn = PyramidFeatures2(filter_list[2], filter_list[3], filter_list[4])

        self.regress = nn.Sequential(
            nn.BatchNorm1d(filter_list[4]*4*3),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4]*4*3, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )
        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 4 * 3),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 4 * 3, 512),
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
        t0 =time()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        x2 = self.conv4(out)
        out = self.conv5(x2)
        x3 = self.conv6(out)
        out = self.conv7(x3)
        fmap = self.conv8(out)

        t1 =time()
        # generate mask
        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            mask = self.mask(features)

        t2 =time()
        # regress

        mask2=self.conv10(mask)
        vec = self.regress(mask2.view(mask2.size(0), -1))

        t3 =time()

        fmap_mask = fmap * mask
        fmap_mask=self.conv9(fmap_mask)

        fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))

        t4 =time()
        fmap=self.conv9(fmap)
        fc = self.fc(fmap.view(fmap.size(0), -1))
        # print( f"trunk time: {t1-t0:.5f} \n"
        #        f"mask_fpn time: {t2 - t1:.5f} \n"
        #        f"mask_fc time: {t3 - t2:.5f} \n"
        #        f"FR_fc time: {t4 - t3:.5f} \n"
        #        )

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

def FaceMobileNet_Occ_L(num_mask=101):
    # filter_list = [64, 64, 128, 256, 512, num_mask]
    is_gray = False
    filter_list = [32, 32, 64, 128, 256, num_mask]
    model = MobileNet_Occ_L(filter_list, is_gray)
    return model
