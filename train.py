import os
import os.path as osp
import torch
import torch.nn as nn

from tqdm import tqdm

from model import FaceMobileNet, ResIRSE, FMask_R, FMask_M, FaceNet, FMask_M_L, FMask_M_qua
from model.metric import ArcFace, CosFace, triplet_loss
from model.loss import FocalLoss
from dataloader.dataset import load_data,WebFace_OCC_LMDB
from config import config as conf
from test import test, test_lfw
import numpy as np
import dataloader.utils  as utils
import json

from tensorboardX import SummaryWriter
import time
import shutil
def main():
    logger, final_output_dir, tb_log_dir = utils.create_logger(conf, 'train')

    # Data Setup
    # dataloader, class_num = load_data(conf, training=True)
    # embedding_size = conf.embedding_size
    device = conf.device
    LMDB = WebFace_OCC_LMDB(db_path = conf.LMDB_FILE,
                               img_size = conf.input_shape[1:], pattern=conf.S,
                               transform=conf.train_transform)
    class_num = LMDB.get_classnum()

    dataloader = torch.utils.data.DataLoader(
        dataset=LMDB,
        batch_size=conf.train_batch_size,
        shuffle=conf.SHUFFLE,
        num_workers=conf.num_workers,
        pin_memory=True)
    # Network Setup
    logger.info('length of train Database: ' + str(len(dataloader.dataset)) + '  Batches: ' + str(len(dataloader)))
    logger.info('Number of Identities: ' + str(class_num))

    # [ARC-R, ARC-M, FMask-R, FMask-M,FaceNet]
    if conf.backbone == 'ARC-R':
        net = ResIRSE(conf.embedding_size, conf.drop_ratio).to(device)
    elif conf.backbone == 'ARC-M':
        net = FaceMobileNet(conf.embedding_size).to(device)
    elif conf.backbone == 'FMask-R':
        net = FMask_R(conf.NUM_MASK).to(device)
    elif conf.backbone == 'FMask-M':
        net = FMask_M(conf.NUM_MASK).to(device)
    elif conf.backbone == 'FMask-M_L':
        net = FMask_M_L(conf.NUM_MASK).to(device)
    elif conf.backbone == 'FMask-M_qua':
        net = FMask_M_qua(conf.NUM_MASK).to(device)
    elif conf.backbone == 'FaceNet':
        net = FaceNet(backbone='', num_classes=class_num, pretrained=False).to(device)


    if conf.metric == 'arcface':
        metric = ArcFace(conf.embedding_size, class_num).to(device)
    else:
        metric = CosFace(conf.embedding_size, class_num).to(device)

    # Training Setup
    criterion_F = FocalLoss(gamma=2)

    criterion_C = nn.CrossEntropyLoss()

    if conf.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}],
                                    lr=conf.lr, momentum=conf.MOMENTUM, weight_decay=conf.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}], lr=conf.lr,)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

    # Checkpoints Setup
    checkpoints = conf.checkpoints
    os.makedirs(checkpoints, exist_ok=True)

    if conf.restore:
        weights_path = osp.join(checkpoints, conf.restore_model)
        net.load_state_dict(torch.load(weights_path, map_location=device))
        metric.load_state_dict(torch.load(osp.join(checkpoints, "metric.pth"), map_location=device))

    # Start training
    net_type = conf.net_type
    net.train()
    for e in range(conf.epoch):
        time_curr = time.time()
        loss=0.0
        loss_cls=0.0
        loss_pred=0.0
        loss_display = 0.0
        loss_cls_dis = 0.0
        loss_pred_dis = 0.0
        start = time.time()
        best_acc = 0.0
        best_keep = [0, 0]
        best_model = False
        for batch_idx, data in tqdm(enumerate(dataloader), desc=f"Epoch {e}/{conf.epoch}",
                                 ascii=True, total=len(dataloader)):
            img, labels, mask_label, imgPaths = data
            if net_type == 0:
                img, labels = img.cuda(), labels.cuda()
                fc = net(img)
                thetas = metric(fc, labels)
                loss = criterion_F(thetas, labels)
            elif net_type == 1:
                img, labels = img.cuda(), labels.cuda()
                mask_label = mask_label.cuda()
                fc_mask, mask, vec, fc = net(img)

                thetas = metric(fc_mask, labels)
                loss_cls = criterion_F(thetas, labels)

                loss_pred = criterion_C(vec, mask_label)
                preds = vec.cpu().detach().numpy()
                preds = np.argmax(preds, axis=1)


                loss = loss_cls + conf.WEIGHT_PRED * loss_pred

                loss_cls_dis += loss_cls.item()
                loss_pred_dis += loss_pred.item()

            else:
                img, labels = img.cuda(), labels.cuda()
                fc0, fc = net.forward_feature(img)
                output = net.forward_classifier(fc0)

                t_loss = triplet_loss()(output, conf.train_batch_size)
                CE_loss = nn.NLLLoss()(nn.functional.log_softmax(output, dim=-1), labels)
                loss = t_loss + CE_loss



            loss_display += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = e * len(dataloader) + batch_idx

            if iters% conf.PRINT_FREQ ==0 and iters != 0:

                output = thetas.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = labels.data.cpu().numpy()
                acc_train = np.mean((output == label).astype(int))

                time_used = time.time() - time_curr
                if batch_idx < conf.PRINT_FREQ:
                    num_freq = batch_idx + 1
                else:
                    num_freq = conf.PRINT_FREQ

                speed = num_freq / time_used
                loss_display /= num_freq
                loss_cls_dis /= num_freq
                loss_pred_dis /= num_freq

                INFO = ' Margin: {:.2f}, Scale: {:.2f}'.format(metric.m, metric.s)
                logger.info(
                    'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, Loss: {:.6f}, Acc_train: {:.4f}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
                        e, batch_idx, len(dataloader), 100. * batch_idx / len(dataloader),
                        iters, loss_display, acc_train, time_used, speed) + INFO)
                logger.info('Cls Loss: {:.4f}; Pred Loss: {:.4f}*{}'.format(loss_cls_dis, loss_pred_dis, conf.WEIGHT_PRED))

                with SummaryWriter(tb_log_dir) as sw:
                    sw.add_scalar('TRAIN_LOSS', loss_display, iters)
                    sw.add_scalar('CLS_LOSS', loss_cls_dis, iters)
                    sw.add_scalar('PRED_LOSS', loss_pred_dis, iters)
                time_curr = time.time()
                loss_display = 0.0
                loss_cls_dis = 0.0
                loss_pred_dis = 0.0

        with torch.no_grad():
            acc, acc_oc = test_lfw(net, tb_log_dir=tb_log_dir, epoch=e)
            # acc = test(net, conf.LFW_PATH, conf.LFW_PAIRS,tb_log_dir, e,'clean')
            # acc_oc = test(net, conf.LFW_OCC_PATH, conf.LFW_PAIRS,tb_log_dir, e,'occ')
        scheduler.step()

        if acc_oc > best_acc:
            best_acc = acc_oc
            best_keep = [acc, acc_oc]
            best_model = True
        else:
            best_model = False

        logger.info('current best accuracy {:.5f}'.format(best_acc))
        logger.info('saving checkpoint to {}'.format(final_output_dir))
        utils.save_checkpoint({
            'epoch': e + 1,
            'model': conf.backbone,
            'state_dict': net.state_dict(),
            'perf': acc_oc,
            'optimizer': optimizer.state_dict(),
            'classifier': metric.state_dict(),
        }, best_model, final_output_dir)
    # save best model with its acc
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    shutil.move(os.path.join(final_output_dir, 'model_best.pth.tar'),
                os.path.join(final_output_dir, 'model_best_{}_{:.4f}_{:.4f}.pth.tar'.format(time_str, best_keep[0], best_keep[1])))

    end = time.time()
    time_used = (end - start) / 3600.0
    logger.info('Done Training, Consumed {:.2f} hours'.format(time_used))

if __name__ == '__main__':
    main()
