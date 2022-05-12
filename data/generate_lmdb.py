import os
import time
import six
import lmdb
import cv2
import numpy  as np
import pyarrow as pa
from PIL import Image
from tqdm import tqdm

import lib.core.utils as utils
import sys
import pdb



def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def imglist_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            # imgPath, imgPath_occ, label = line.strip().split(' ')
            imgPath, imgPath_occ, label = line.split(' ')
            imgList.append((imgPath, imgPath_occ, int(label)))
    return imgList

def generate_file_list(root, write_path):
    file_list = []
    for i, class_name in enumerate(sorted(os.listdir(root))):
        for img_name in sorted(os.listdir(os.path.join(root, class_name))):
            path = os.path.join(class_name, img_name) + ' ' + str(i) + '\n'
            print(path)
            file_list.append(path)
    print(len(file_list))
    with open(write_path, 'w') as f:
        f.writelines(file_list)

def checkdir(path):
    dirname = os.path.dirname(path)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

def buf2img(imgbuf):
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img

def img2buf(img):
    imgbuf = six.BytesIO()
    img.save(imgbuf, format='jpeg')
    imgbuf = imgbuf.getvalue()
    return imgbuf

def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

if __name__ == '__main__':
    S = int(sys.argv[1])
    # imglist is like '00045/001.jpg'
    # occlist is like 'cup/0093.png'
    # generate_file_list(root='data/datasets/faces_webface_112x112/images/', write_path='data/datasets/faces_webface_112x112/images.txt')

    # S=3
    WebFace_LMDB = './datasets/Webface_112x96_S'+str(S)+'.lmdb'
    WebFace_Images = './datasets/Webface-OCC-112x96'
    WebFace_List = 'Webface_occ_train-112x96.txt'

    imgList = imglist_reader(WebFace_List)

    checkdir(WebFace_LMDB)
    lmdb_path = WebFace_LMDB
    isdir = os.path.isdir(lmdb_path)

    img_size = (112, 96)
    grids = utils.get_grids(*img_size, S)
    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, item in tqdm(enumerate(imgList), total=len(imgList)):
        imgPath, imgPath_occ, label = item

        img = raw_reader(os.path.join(WebFace_Images, imgPath))
        img_occ = raw_reader(os.path.join(WebFace_Images, imgPath_occ))
        # pdb.set_trace()

        #get label
        img_buf = Image.open(os.path.join(WebFace_Images, imgPath))
        img_occ_buf = Image.open(os.path.join(WebFace_Images, imgPath_occ))
        img_np = np.array(img_buf.convert('L'))
        img_occ_np = np.array(img_occ_buf.convert('L'))
        mask_np=img_np-img_occ_np
        error = 5
        for i in range(mask_np.shape[0]):
            for j in range(mask_np.shape[1]):
                if (mask_np[i][j] > error and mask_np[i][j] < 255 - error):
                    mask_np[i][j] = 1
                else:
                    mask_np[i][j] = 0

        mask = cv2.dilate(mask_np, (3,3))
        # cal mask label
        mask_label = utils.cal_similarity_label(grids, mask)

        txn.put(u'{}'.format(idx*2).encode('ascii'), dumps_pyarrow((img, label, 0, imgPath)))
        txn.put(u'{}'.format(idx*2+1).encode('ascii'), dumps_pyarrow((img_occ, label, mask_label, imgPath_occ)))

        # byteflow = txn.get(b'0')
        # unpacked = pa.deserialize(byteflow)
        # imgbuf, img_occbuf, label, imgPath, imgPath_occ  = unpacked
        #
        # buf = six.BytesIO()
        # buf.write(imgbuf)
        # buf.seek(0)
        # img = Image.open(buf).convert('RGB')
        # img.save('temp.jpg')


        if 2 * idx % 5000 == 0:
            # time_cur = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
            # print('{}: {}/{}'.format(time_cur, idx, len(imgList)))
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    idx = idx * 2 + 1
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    print(len(keys))
    print(len(imgList))
    db.sync()
    db.close()
