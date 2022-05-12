import os
import random

father_path='./datasets/Webface-OCC-112x96'
f = 'Webface_occ_train-112x96.txt'
names = os.listdir(father_path)
i=0


all_num=len(names)
num=0
with open(f,"w") as file:
    i=0
    for name in names:
    # if i<20:
        name_path=os.path.join(father_path, name)
        pics = os.listdir(name_path)
        print(f"num{i}  all{all_num}")
        for pic in pics:
            all=pic.split('_')
            if len(all)>1:
                num+=1
                face1=all[0]+'.jpg'
                a=os.path.join(name, face1)
                b=os.path.join(name, pic)
                file.write(a + " " + b + " "+str(i)+"\n")
        i+=1
    # print(num)


