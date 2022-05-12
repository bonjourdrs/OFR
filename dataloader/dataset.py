import os
import six
import lmdb
import torch
import pickle
import random
import numpy as np
import pyarrow as pa
import cv2
from torch.utils.data import Dataset
from PIL import Image, ImageFile, ImageDraw
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from config import config as conf
import dataloader.utils as utils
import time


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

def load_data(conf, training=True):
    if training:
        dataroot = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_batch_size
    else:
        dataroot = conf.test_root
        transform = conf.test_transform
        batch_size = conf.test_batch_size

    data = ImageFolder(dataroot, transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, 
        pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num


class WebFace_OCC_LMDB(Dataset):
    def __init__(self, db_path, img_size=(112, 96), pattern=5, transform=None):
        super(WebFace_OCC_LMDB, self).__init__()
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))

        self.grids = utils.get_grids(*img_size, pattern)
        self.img_size = img_size

        self.transform = transform

    def PIL_reader(self, path):
        try:
            with open(path, 'rb') as f:
                return Image.open(f).convert('RGB')
        except IOError:
            print('Cannot load image ' + path)

    def buf2img(self, imgbuf):
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    def img2buf(self, img):
        imgbuf = six.BytesIO()
        img.save(imgbuf, format='jpeg')
        imgbuf = imgbuf.getvalue()
        return imgbuf

    def occlude_img(self, img):
        # occlude img
        occPath = random.choice(self.occList)
        occ = self.PIL_reader(os.path.join(self.occRoot, occPath))
        factor = random.choice(np.linspace(1, 5, 9, endpoint=True))
        img_occ, mask, _ = utils.occluded_image_ratio(img.copy(), occ, factor)

        # cal mask label
        mask_label = utils.cal_similarity_label(self.grids, mask)
        return img_occ, mask_label, mask

    def get_classnum(self):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[self.length-1])
        return int(pa.deserialize(byteflow)[1] + 1)

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        imgbuf, label, mask_label, imgPath  = unpacked

        # load image
        img = self.buf2img(imgbuf)
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = random.choice([img, img_flip])

        img = self.transform(img)
            # img_occ = self.transform(img_occ)
        return img, label, mask_label, imgPath

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class LFW_Image(Dataset):
    def __init__(self, conf):
        super(LFW_Image, self).__init__()
        self.lfw_path = conf.LFW_PATH
        self.lfw_occ_path = conf.LFW_OCC_PATH
        self.pairs = self.get_pairs_lines(conf.LFW_PAIRS)
        self.image_size = conf.input_shape[1:]

        self.transform = conf.test_transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[1:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split(' ')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")


            # if os.path.exists(self.lfw_occ_path + name1) and os.path.exists(self.lfw_occ_path + name2):
            if os.path.exists(self.lfw_occ_path + name1) and os.path.exists(self.lfw_occ_path + name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        p = self.pairs[index]

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")

        with open(self.lfw_path + name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')
            img1 = letterbox_image(img1,self.image_size)

        with open(self.lfw_path + name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')
            img2 = letterbox_image(img2,self.image_size)

        with open(self.lfw_occ_path + name2, 'rb') as f:
            img2_occ =  Image.open(f).convert('RGB')
            img2_occ = letterbox_image(img2_occ,self.image_size)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img2_occ = self.transform(img2_occ)
        return img1, img2, img2_occ, sameflag

    def __len__(self):
        return len(self.pairs)

class GFRTD(Dataset):
    def __init__(self, type='star_image1'):
        super(GFRTD, self).__init__()
        if type.find('random')<0:
            self.path = '../../../dataset/3_dataset/star_image/'+type
        else:
            self.path = '../../../dataset/3_dataset/star_image/random/'+type

        pairs_path = "../../../dataset/3_dataset/all_list2/" + type + ".txt"
        self.pairs = self.get_pairs_lines(pairs_path)
        self.image_size = conf.input_shape[1:]
        self.transform = conf.test_transform
        self.valid_check()
        self.num_pairs = len(self.pairs)

    def get_pairs_lines(self, path):
        with open(path) as f:
            pairs_lines = f.readlines()[1:]
        return pairs_lines

    def valid_check(self):
        valid_pairs = []
        for pair in self.pairs:
            p = pair.replace('\n', '').split(' ')

            name1 =self.path +'/'+ p[0]
            name2 =self.path +'/'+ p[1]

            if os.path.exists(name1) and os.path.exists(name2):
                valid_pairs.append(p)

        print('valid rate: {:.4f}'.format(len(valid_pairs) / len(self.pairs)))

        self.pairs = valid_pairs

    def __getitem__(self, index):
        p = self.pairs[index]
        sameflag=int(p[2])

        name1 = self.path + '/' + p[0]
        name2 = self.path + '/' + p[1]
        with open(name1, 'rb') as f:
            img1 =  Image.open(f).convert('RGB')
            img1 = letterbox_image(img1,self.image_size)

        with open(name2, 'rb') as f:
            img2 =  Image.open(f).convert('RGB')
            img2 = letterbox_image(img2,self.image_size)


        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, sameflag

    def __len__(self):
        return len(self.pairs)

