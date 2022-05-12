# -*- coding: utf-8 -*-
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import config as conf

import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
import glob
from PIL import Image
from sklearn.metrics import roc_curve,roc_auc_score,plot_roc_curve,auc
import matplotlib.pyplot as plt
from config import config as conf
from model import FaceMobileNet, ResIRSE, FMask_R, FMask_M, FaceNet,FMask_M_L
from tqdm import tqdm
from dataloader.dataset import LFW_Image, GFRTD
from tensorboardX import SummaryWriter

def featurize_double(img1,img2,transform, net, device):
    image1=transform(img1)
    image2=transform(img2)
    data1 = image1[:, None, :, :]    # shape: (batch, 1, 128, 128)
    data2 = image2[:, None, :, :]    # shape: (batch, 1, 128, 128)
    data1 = data1.to(device)
    data2 = data2.to(device)
    net = net.to(device)
    fc = net(data1)
    fc_occ = net(data2)
    return fc,fc_occ
def cosine_similarity(f1, f2):
    # compute cosine_similarity for 2-D array
    f1 = f1.detach().numpy()
    f2 = f2.detach().numpy()

    A = np.sum(f1*f2, axis=0)
    B = np.linalg.norm(f1, axis=0) * np.linalg.norm(f2, axis=0) + 1e-5

    return A / B


def letterbox_image(image, size):
    image = image.convert("RGB")
    iw, ih = image.size
    h, w = size
    scale = max(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', [w,h], (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    # image.show()
    # new_image.show()
    # time.sleep(5)
    return new_image

def scan_video(net,transform, test_video,device):
    simlirty=np.zeros(128)
    f = '../../../dataset/2_wear_occ/data_video/陈坤/test_net/input.jpg'

    img1 = Image.open(f).convert('RGB')
    img1 = letterbox_image(img1, conf.input_shape[1:])
    image1 = transform(img1)
    data1 = image1[None, :, :, :]  # shape: (batch, 1, 128, 128)
    data1 = data1.to(device)
    net = net.to(device)
    fc, _, _, _ = net(data1)
    fc = fc.to('cpu').squeeze()

    for i in range(128):
        name='8'+'_'+str(i)+'.jpg'
        pic=os.path.join(test_video,name)
        pic = Image.open(pic).convert('RGB')
        pic = letterbox_image(pic, conf.input_shape[1:])
        image1 = transform(pic)
        data1 = image1[None, :, :, :]  # shape: (batch, 1, 128, 128)
        data1 = data1.to(device)
        net = net.to(device)
        fc_mask, _, _, _ = net(data1)
        fc_mask = fc_mask.to('cpu').squeeze()
        # print(fc_mask)
        # print(fc)


        distance = cosine_similarity(fc_mask, fc)
        k=2
        distance = distance/k +(k-1)/k
        simlirty[i]=(distance)
        print(i)



    # y = simlirty
    for i in range(128):
        print(i)
        plt.clf()
        plt.ylim(0,1)
        plt.xlabel('time')
        plt.ylabel('occlusion')
        plt.grid(linestyle='-.')
        x = np.linspace(0, i, i+1)
        y=simlirty[0:i+1]
        plt.plot(x, y, ls='-', lw=2, label='occ', color='purple')
        plt.legend()
        plt.savefig(test_video+'/'+str(i)+'.jpg')
    # plt.show()

def get_sim(model,transform, pair_list, test_root,device):
    # with open(pair_list, 'r',encoding='gbk') as f:
    with open(pair_list, 'r') as f:
        pairs = f.readlines()
    cos_sim = nn.CosineSimilarity()
    similarities = []
    labels = []
    i=0
    for pair in pairs:
        i+=1
        if i%20==0:
            print(i)
        img1, img2, label = pair.split()
        img1=img1.replace('\\', '/')
        img2=img2.replace('\\', '/')
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        image1 = Image.open(img1)
        image2 = Image.open(img2)
        feature1,feature2=featurize_double(image1, image2, transform, model, device)
        print(feature1)
        print(feature2)
    return 0



if __name__ == '__main__':
    device = conf.device
    # backbone = 'FMask-R'  # [ARC-R, ARC-M, FMask-R, FMask-M,FaceNet]
    backbone = conf.backbone
    if backbone == 'ARC-R':
        net_type=0
        model = ResIRSE(conf.embedding_size, conf.drop_ratio).to(device)
    elif backbone == 'ARC-M':
        net_type=0
        model = FaceMobileNet(conf.embedding_size).to(device)
    elif backbone == 'FMask-R':
        net_type=1
        model = FMask_R(conf.NUM_MASK).to(device)
    elif backbone == 'FMask-M':
        net_type=1
        model = FMask_M(conf.NUM_MASK).to(device)
    elif backbone == 'FMask-M_L':
        net_type=1
        model = FMask_M_L(conf.NUM_MASK).to(device)
    elif backbone == 'FaceNet':
        net_type=2
        model = FaceNet(backbone='mobilenet', pretrained=False,mode='predicted').to(device)

    # model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
    model = nn.DataParallel(model).cuda()
    # weight = torch.load(conf.test_model, map_location=conf.device)
    # dict = model.state_dict()

    state_dict = torch.load(conf.test_model)
    state_dict = state_dict['state_dict']

    if isinstance(model, torch.nn.DataParallel):
    # if net_type == 1:
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    test_video = '../../../dataset/2_wear_occ/data_video/陈坤/test_net'

    scan_video(model,conf.test_transform, test_video,conf.device)


