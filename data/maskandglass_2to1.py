import os
from tqdm import tqdm
import glob
import shutil
import random
father_path = './data/datasets/lfw-occ/lfw-112X96'
glass_path = './data/datasets/lfw-occ/lfw-112X96_glasses'
mask_path = './data/datasets/lfw-occ/lfw-112X96_masked'
save_path = './data/datasets/lfw-occ/lfw-112X96_occ_same'
names = os.listdir(father_path)
for name in tqdm(names):
    name_path = os.path.join(father_path,name)
    pics = os.listdir(name_path)
    glass_name_path = os.path.join(glass_path,name)
    mask_name_path = os.path.join(mask_path,name)
    save_name_path = os.path.join(save_path,name)
    if not os.path.exists(save_name_path):
        os.mkdir(save_name_path)
    for num,pic in enumerate(pics):
        saver = os.path.join(save_name_path,pic)
        g_path = os.path.join(glass_name_path,pic)
        g = g_path.split('.jpg')
        g_path= g[0]+'*'
        g_all = glob.glob(g_path)

        m_path = os.path.join(mask_name_path,pic)
        m = m_path.split('.jpg')
        m_path= m[0]+'*'
        m_all = glob.glob(m_path)
        all=[]
        if len(g_all)>0:
            all.append(g_all[0])
        if len(m_all)>0:
            all.append(m_all[0])

        if len(all) > 0:
            shutil.copy(random.choice(all),saver)

        print(1)
        # save_path = os.path.join(name_path,name+'_'+ '{:04}.jpg'.format(num))
        # os.rename(pic_path,save_path)
