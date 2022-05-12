import face_recognition
from config import config as conf
import torch
import os
from torch.utils.data import Dataset
import numpy as np
# import tqdm
from tqdm import tqdm

def compute_distance(img1, img2, flag):
    print(img1)
    print(img2)
    # I1 = face_recognition.load_image_file(img1[0])
    # I1 = face_recognition.face_encodings(I1)[0]
    I2 = face_recognition.load_image_file(img2[0])
    I2 = face_recognition.face_encodings(I2)[0]
    # print(my_face_encoding.shape)
    # print(unknown_face_encoding.shape)
    distance = np.linalg.norm(I2 - I2, axis=0)
    # distance = face_recognition.face_distance(my_face_encoding,unknown_face_encoding)
    # distance = face_recognition.compare_faces([my_face_encoding],unknown_face_encoding)
    print(distance)
    return [distance, flag]

class test_Image(Dataset):
    def __init__(self):
        super(test_Image, self).__init__()
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
            sameflag = 1.
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0.
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")

        return self.lfw_path + name1, self.lfw_path + name2, self.lfw_occ_path + name2, sameflag

    def __len__(self):
        return len(self.pairs)

LFW_test_loader = torch.utils.data.DataLoader(
    test_Image(),
    batch_size=1,
    shuffle=conf.Test_SHUFFLE,
    num_workers=conf.num_workers,
    pin_memory=True)

predicts = np.zeros(shape=(len(LFW_test_loader.dataset), 2))
predicts_occ = np.zeros(shape=(len(LFW_test_loader.dataset), 2))
cur = 0

from sklearn.metrics import roc_curve,roc_auc_score,plot_roc_curve,auc

for batch_idx, (img1, img2, img2_occ, flag) in tqdm(enumerate(LFW_test_loader), desc="test_start", ascii=True,
                                                    total=len(LFW_test_loader)):
    predicts[cur:cur + 1] = compute_distance(img1, img2, flag)
    predicts_occ[cur:cur + 1] = compute_distance(img1, img2_occ, flag)
    cur += flag.shape[0]
# fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)  ###计算auc的值

fpr, tpr, threshold  = roc_curve(predicts[:, 1], predicts[:, 0])
roc_auc = auc(fpr, tpr)  ###计算auc的值
fpr, tpr, threshold  = roc_curve(predicts_occ[:, 1], predicts_occ[:, 0])
roc_auc_occ = auc(fpr, tpr)  ###计算auc的值


auc_occ = roc_curve(predicts_occ[:, 1], predicts_occ[:, 0])
# accuracy = obtain_acc(predicts, test_loader.dataset.num_pairs)
# accuracy_occ = obtain_acc(predicts_occ, test_loader.dataset.num_pairs)



# Now we can see the two face encodings are of the same person with `compare_faces`!
#
# import Image
# import face_recognition
# image = face_recognition.load_image_file('F:/Python27/Scripts/all.jpg')
# face_locations = face_recognition.face_locations(image)
#
# #face_locations =face_recognition.
#
# #face_locations(image,number_of_times_to_upsample=0,model='cnn')
# print('i found {} face(s) in this photograph.'.format(len(face_locations)))
# for face_location in face_locations:
#     top,right,bottom,left = face_location
#     print('A face is located at pixel location Top:{},Left:{},Bottom:{},Right:{}'.format(top,right,bottom,left))
#     face_image = image[top:bottom,left:right]
#     pil_image=Image.fromarray(face_image)
#     pil_image.show()