import os
import random

# from config import config as conf

# father_path='./datasets/lfw-occ/lfw-112X96_occ'
father_path='./datasets/star_image/star_image1'
name_paths = os.listdir(father_path)
i=0

f='./datasets/star_image/'+father_path.split('/')[-1]+'.txt'
# f = './datasets/lfw-occ/lfw_112x96_occ_pairs.txt'


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

            num1=face1.split('.')[0].split('_')[-1]
            num2=face2.split('.')[0].split('_')[-1]
            file.write(name_path + " " + str(int(num1))+ " " + diff + ' ' + str(int(num2))+"\n")
            # file.write(a + " " + b + " 0"+"\n")

        if len(pic_paths)>1:
            for i in range(1):
                face3=random.choice(pic_paths)
                face4=random.choice(pic_paths)
                while face3==face4:
                    face4 = random.choice(pic_paths)
                c=os.path.join(name_path,face3)
                d=os.path.join(name_path,face4)
                num3=face3.split('.')[0].split('_')[-1]
                num4=face4.split('.')[0].split('_')[-1]
                file.write(name_path + " " + str(int(num3))+ " "+ str(int(num4))+"\n")
                # file.write(c + " " + d + " 1"+"\n")

