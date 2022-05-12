import os
from tqdm import tqdm
father_path = './datasets/lfw-occ/lfw-112X96_masked2'
# from glob import glob
# pic=glob('*.jpg')
# for p in pic:
#     os.rename(p,os.path.join(father_path,p))
# print(len(pic))
names = os.listdir(father_path)
for name in tqdm(names):
    name_path = os.path.join(father_path,name)
    pics = os.listdir(name_path)
    for num,pic in enumerate(pics):
        pic_path = os.path.join(name_path,pic)
        # temp = pic.split('_glasses')[0] + '.jpg'
        temp = pic[0:pic.find('0') + 4]+ '.jpg'
        save_path = os.path.join(name_path,temp)
        # save_path = os.path.join(name_path,name+'_'+ '{:04}.jpg'.format(num))
        os.rename(pic_path,save_path)
