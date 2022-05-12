# -*- coding: utf-8 -*-
import os
import os.path as osp
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from config import config as conf
from model import FaceMobileNet



def unique_image(pair_list) -> set:
    """Return unique image path in pair_list.txt"""
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split()
        unique.add(id1)
        unique.add(id2)
    return unique


def group_image(images, batch):
    """Group image paths by batch size"""
    # images = list(images)
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


def featurize(images: list, transform, net, device) -> dict:
    """featurize each image and save into a dictionary
    Args:
        images: image paths
        transform: test transform
        net: pretrained model
        device: cpu or cuda
    Returns:
        Dict (key: imagePath, value: feature)
    """
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data) 
    res = {img: feature for (img, feature) in zip(images, features)}
    return res


def featurize_singal(img, transform, net, device):
    image=transform(img)
    data = image[:, None, :, :]    # shape: (batch, 1, 128, 128)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data)
    return features

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
    with open(pair_list, 'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()
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

def get_similarity(feature_dict, img_f):
    max_similarity=0
    for name in feature_dict:
        temp=feature_dict[name]
        similarity=cosin_metric(temp,img_f[0])
        if similarity>max_similarity:
            max_similarity=similarity
            out=name
    return max_similarity,out


if __name__ == '__main__':

    model = FaceMobileNet(conf.embedding_size)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
    model.eval()


    image_path = "./input/drs.jpg"
    img_raw = Image.open(image_path)
    names=os.listdir(conf.test_root)
    images=[]
    for name in names:
        name_all=osp.join(conf.test_root, name)
        frame=os.listdir(name_all)
        images.append(osp.join(name_all,frame[0]))


    groups = group_image(images, conf.test_batch_size)

    feature_dict = dict()
    i=0
    img_f=featurize_singal(img_raw,conf.test_transform,model,conf.device)
    for group in groups:
        d = featurize(group, conf.test_transform, model, conf.device)
        feature_dict.update(d)

    max_similarity,name = get_similarity(feature_dict, img_f)

    print(f"max_similarity: {max_similarity}\n")
    print(name)
