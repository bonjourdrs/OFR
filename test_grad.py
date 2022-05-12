# -*- coding: utf-8 -*-
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from config import config as conf
from model import FaceMobileNet,ResIRSE
from grad_cam import ShowGradCam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
def unique_image(pair_list) -> set:
    """Return unique image path in pair_list.txt"""
    # with open(pair_list, 'r',encoding='gbk') as fd:
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split()
        id1=id1.replace('\\', '/')
        id2=id2.replace('\\', '/')
        unique.add(id1)
        unique.add(id2)
    return unique


def group_image(images: set, batch) -> list:
    """Group image paths by batch size"""
    images = list(images)
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch + i, size)
        res.append(images[i : end])
    return res


def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]    # shape: (batch, 1, 128, 128)
    return data





def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


def compute_accuracy(feature_dict, pair_list, test_root):
    # with open(pair_list, 'r',encoding='gbk') as f:
    with open(pair_list, 'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()
        img1=img1.replace('\\', '/')
        img2=img2.replace('\\', '/')
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold


if __name__ == '__main__':
    if conf.backbone == 'resnet':
        model = ResIRSE(conf.embedding_size, conf.drop_ratio)
    else:
        model = FaceMobileNet(conf.embedding_size)
    # cam_lay=model.flatten
    # gradcam_lay = [model.flatten]
    # model.conv10.net.
    gradcam=ShowGradCam(model.flatten)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
    model.eval()

    # cam = GradCAM(model=model,target_layers=gradcam_lay, use_cuda=True)
    res = []
    im = Image.open("./input/drs.jpg")
    im = conf.test_transform(im)
    res.append(im)
    data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]    # shape: (batch, 1, 128, 128)
    data = data.to(conf.device)
    net = model.to(conf.device)
    with torch.no_grad():
        features = net(data)
    # res = {img: feature for (img, feature) in zip(images, features)}
    # return res
    # backward
    # net.zero_grad()
    # class_loss = torch.sum(features)
    # class_loss.backward()
    # gray = cam(input_tensor=data[0], target_category=1)
    # g=gray[0,:]
    # input_img = cv2.imread("./input/drs.jpg")
    # v=show_cam_on_image(input_img,g,use_rgb=True)
    gradcam.show_on_img("./input/drs.jpg")
    print("down")



