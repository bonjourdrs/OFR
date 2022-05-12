import os
import random

from config import config as conf

father_path=conf.test_root
name_paths = os.listdir(father_path)
i=0

f = conf.test_list_temp


with open(f,"w") as file:
    for name_path in name_paths:

        pic_paths = os.listdir(os.path.join(father_path, name_path))


        for i in range(1):
            diff = name_path
            while diff == name_path:
                diff = random.choice(name_paths)
            diff_pic_paths = os.listdir(os.path.join(father_path, diff))
            face1=random.choice(pic_paths)
            face2=random.choice(diff_pic_paths)
            a=os.path.join(name_path,face1)
            b=os.path.join(diff,face2)
            file.write(a + " " + b + " 0"+"\n")

        for i in range(1):
            face3=random.choice(pic_paths)
            face4=random.choice(pic_paths)
            while face3==face4:
                face4 = random.choice(pic_paths)
            c=os.path.join(name_path,face3)
            d=os.path.join(name_path,face4)
            file.write(c + " " + d + " 1"+"\n")

