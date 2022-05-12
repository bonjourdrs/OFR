import os
import cv2
from tqdm import tqdm

father_path='./datasets/Webface-OCC-112x112'
son_path='./datasets/Webface-OCC-112x96'

names = os.listdir(father_path)
i=0
for name in tqdm(names,total=len(names)):
    name_path = os.path.join(father_path, name)
    pics = os.listdir(name_path)
    name_path_save = os.path.join(son_path, name)
    if not os.path.exists(name_path_save):
        os.mkdir(name_path_save)
    for pic in pics:
        source = os.path.join(name_path,pic)
        save =os.path.join(name_path_save,pic)
        i += 1
        img = cv2.imread(source)
        cv2.imwrite(save,img[:,8:104])


print(i)
