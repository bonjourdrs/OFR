# -*- coding: utf-8 -*-
import os
import os.path as osp
import time
from torch2trt import torch2trt
import torch
import torch.nn as nn
import numpy as np
import glob
from PIL import Image
from sklearn.metrics import roc_curve,roc_auc_score,plot_roc_curve,auc
import matplotlib.pyplot as plt
from config import config as conf
from model import FaceMobileNet, ResIRSE, FMask_R, FMask_M, FaceNet,FMask_M_L,FMask_M_qua
from tqdm import tqdm
from dataloader.dataset import LFW_Image, GFRTD
from tensorboardX import SummaryWriter
def unique_image(pair_list) -> set:
    # with open(pair_list, 'r',encoding='gbk') as fd:
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        # id1, id2, _ = pair.split()
        p = pair.replace('\n', '').split(' ')
        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        id1=name1.replace('\\', '/')
        id2=name2.replace('\\', '/')
        unique.add(id1)
        unique.add(id2)
    return unique

def group_image(images: set, batch) -> list:
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
    data = torch.stack(res,dim=0)
    # data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    # data = data[:, :, :, :]    # shape: (batch, 1, 128, 128)
    return data

def featurize(images: list, transform, net, device) -> dict:
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        if conf.net_type==0:
            features = net(data)
        elif conf.net_type==1:
            features, _, _, _ = net(data)
    res = {img: feature for (img, feature) in zip(images, features)}
    return res

def featurize_tsne(images: list, transform, net, net_type, device) -> dict:
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        if net_type==0:
            features = net(data)
        elif net_type==1:
            features, _, _, _ = net(data)
    # res = {img: feature for (img, feature) in zip(images, features)}
    return features.cpu().numpy()

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosine_similarity(f1, f2):
    # compute cosine_similarity for 2-D array
    f1 = f1.numpy()
    f2 = f2.numpy()

    A = np.sum(f1*f2, axis=1)
    B = np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1) + 1e-5

    return A / B


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

def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    # plt.xscale('log')
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()
    return roc_auc

def compute_accuracy(feature_dict, pair_list, test_root, ROC):
    # with open(pair_list, 'r',encoding='gbk') as f:
    with open(pair_list, 'r') as f:
        pairs = f.readlines()
    similarities = []
    labels = []

    for pair in pairs:
        p = pair.replace('\n', '').split(' ')
        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1=name1.replace('\\', '/')
        img2=name2.replace('\\', '/')
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(sameflag)

        similarity = cosin_metric(feature1, feature2)

        similarities.append(similarity)
        labels.append(label)
    if ROC==1:
        acu_curve(labels,similarities)
    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold

def test(model, test_root, test_list, tb_log_dir=None, epoch=None, mode=None, ROC=0):
    images = unique_image(test_list)
    images = [osp.join(test_root, img) for img in images]
    groups = group_image(images, conf.test_batch_size)

    feature_dict = dict()
    for group in tqdm(groups, desc="test_start", ascii=True, total=len(groups)):
        d = featurize(group, conf.test_transform, model, conf.device)
        feature_dict.update(d)
    accuracy, threshold = compute_accuracy(feature_dict, test_list, test_root, ROC)


    if mode == 'clean':
        writer = SummaryWriter(tb_log_dir)
        writer.add_scalar('CLEAN_LFW_ACC', np.mean(accuracy), epoch)
        writer.close()
    else:
        writer = SummaryWriter(tb_log_dir)
        writer.add_scalar('OCC_LFW_ACC', np.mean(accuracy), epoch)
        writer.close()
    print(
        f"Test path: {test_root} " f"Test list: {test_list}\n"
        f"Accuracy: {accuracy:.5f} " f"Threshold: {threshold:.5f}\n"
    )
    return accuracy



def extractDeepFeature(img, model):
    img = img.to('cuda')
    if conf.net_type==0:
        fc_mask = model(img)
        fc_mask = fc_mask.to('cpu').squeeze()
    elif conf.net_type == 1:
        fc_mask, mask, vec, fc = model(img)
        fc, fc_mask = fc.to('cpu').squeeze(), fc_mask.to('cpu').squeeze()
    else:
        fc_mask = model(img)
        fc_mask = fc_mask.to('cpu').squeeze()
    return fc_mask

def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy

def obtain_acc(predicts, num_class):
    accuracy = []
    thd = []
    folds = KFold(n=num_class, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)

    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    return np.mean(accuracy)


def compute_distance(img1, img2, model, flag):

    f2 = extractDeepFeature(img2, model)
    f1 = extractDeepFeature(img1, model)
    # distance = cosin_metric(f1, f2)
    distance = cosine_similarity(f1, f2)

    flag = flag.squeeze().numpy()
    return np.stack((distance, flag), axis=1)


test_dataloader = torch.utils.data.DataLoader(
    LFW_Image(conf),
    batch_size=conf.test_batch_size,
    shuffle=conf.Test_SHUFFLE,
    num_workers=conf.num_workers,
    pin_memory=True)

def test_lfw(model, test_loader = test_dataloader, tb_log_dir=None, epoch=None):
    model.eval()
    predicts = np.zeros(shape=(len(test_loader.dataset), 2))
    predicts_occ = np.zeros(shape=(len(test_loader.dataset), 2))
    cur=0

    a= time.time()
    with torch.no_grad():
        for batch_idx, (img1, img2, img2_occ, flag) in tqdm(enumerate(test_loader), desc="test_start", ascii=True, total=len(test_loader)):
            predicts[cur:cur+flag.shape[0]] = compute_distance(img1, img2, model, flag)
            predicts_occ[cur:cur+flag.shape[0]] = compute_distance(img1, img2_occ, model, flag)
            cur += flag.shape[0]
    print(time.time()-a,'S')
    auc = acu_curve(predicts[:,1],predicts[:,0] )
    auc_occ = acu_curve(predicts_occ[:,1],predicts_occ[:,0] )
    accuracy = obtain_acc(predicts, test_loader.dataset.num_pairs)
    accuracy_occ = obtain_acc(predicts_occ, test_loader.dataset.num_pairs)
    if tb_log_dir:
        writer = SummaryWriter(tb_log_dir)
        writer.add_scalar('CLEAN_LFW_ACC', np.mean(accuracy), epoch)
        writer.add_scalar('OCC_LFW_ACC', np.mean(accuracy_occ), epoch)
        writer.close()
    print(
        f"Accuracy: {accuracy:.5f} " f"Accuracy_occ: {accuracy_occ:.5f}\n"
        f"AUC: {auc:.5f} " f"AUC_occ: {auc_occ:.5f}\n"
    )

    return accuracy, accuracy_occ


def test_GFRTD(model, test_loader, tb_log_dir=None, epoch=None):
    model.eval()
    predicts = np.zeros(shape=(len(test_loader.dataset), 2))
    cur=0
    with torch.no_grad():
        for batch_idx, (img1, img2, flag) in tqdm(enumerate(test_loader), desc="test_start", ascii=True, total=len(test_loader)):
            predicts[cur:cur+flag.shape[0]] = compute_distance(img1, img2, model, flag)
            cur += flag.shape[0]

    accuracy = obtain_acc(predicts, test_loader.dataset.num_pairs)
    if tb_log_dir:
        writer = SummaryWriter(tb_log_dir)
        writer.add_scalar('GFRTD_ACC', np.mean(accuracy), epoch)
        writer.close()
    print(
        f"Accuracy: {accuracy:.5f} \n"
    )

    return accuracy
def tsne(model,net_type):
    # imgs = glob.glob('./data/datasets/tsne2/*/*.jpg')
    imgs = glob.glob('./tsne6/*/*.jpg')
    groups = group_image(imgs, conf.test_batch_size)

    feature_dict = []
    for group in tqdm(groups, desc="test_start", ascii=True, total=len(groups)):
        d = featurize_tsne(group, conf.test_transform, model, net_type, conf.device)
        feature_dict.append(d)
    # d = featurize(group, conf.test_transform, model, conf.device)
    feature_dict = np.concatenate(feature_dict,axis=0)
    np.save('./tsne5_/'+backbone+'_'+conf.test_model.split('/')[-1].split('.')[0]+'_.npy',feature_dict)
    print(backbone)

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
    elif backbone == 'FMask-M_qua':
        model = FMask_M_qua(conf.NUM_MASK).to(device)
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

    batch_size = 64
    x = torch.rand((batch_size, 3, 112, 96), dtype=torch.float).cuda()
    # convert to TensorRT feeding sample data as input
    # model_trt = torch2trt(model.eval(), [x], max_batch_size=batch_size, int8_mode=True)
    model_trt = torch2trt(model.eval(), [x], max_batch_size=batch_size, int8_mode=False, fp16_mode=True)

    torch.save(model_trt.state_dict(), 'M_L_fp16.pth')
    # torch.save(model_trt.state_dict(), 'M_L_int8.pth')

    model.eval()
    # x = torch.rand((conf.test_batch_size, 3, 112, 96), dtype=torch.float).cuda()
    # model_trt = torch2trt(model.eval(), [x], max_batch_size=conf.test_batch_size)
    # model = model_trt
    # tsne(model, net_type)
    #
    LFW_test_loader = torch.utils.data.DataLoader(
        LFW_Image(conf),
        batch_size=conf.test_batch_size,
        shuffle=conf.Test_SHUFFLE,
        num_workers=conf.num_workers,
        pin_memory=True)
    test_lfw(model,LFW_test_loader)
    # for i in range(20):
    #     num=5*i+5
    #     print(num)
    #     model.eval()
    #     GFRTD_test_loader = torch.utils.data.DataLoader(
    #         GFRTD('star_image1_random'+str(num)),
    #         #[star_image1, star_image1_face, star_image1_ear, star_image1_glasses, star_image1_head
    #         # star_image1_mouse star_image1_nose star_image1_random5 star_image1_random100]
    #         batch_size=conf.test_batch_size,
    #         shuffle=conf.Test_SHUFFLE,
    #         num_workers=conf.num_workers,
    #         pin_memory=True)
    #     test_GFRTD(model, GFRTD_test_loader)

    # test(model, conf.LFW_PATH, conf.LFW_PAIRS, tb_log_dir=None, epoch=None, mode=None, ROC=0)
    # test(model, conf.LFW_OCC_PATH, conf.LFW_PAIRS, tb_log_dir=None, epoch=None, mode=None, ROC=0)
    # test(model, 1)
