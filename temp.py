import os
import glob
imgs = glob.glob('./tsne4/*/*.jpg')
print(imgs[35])
os.remove(imgs[282])
os.remove(imgs[123])
os.remove(imgs[195])
# os.remove(imgs[269])
# os.remove(imgs[22])
# os.remove(imgs[269])
# os.remove(imgs[269])

f = 'data/datasets/lfw-occ/lfw-112X96_occ/'
# names = os.listdir(f)
# for name in names:
#     name_p = os.path.join(f,name)
#     if len(os.listdir(name_p)) > 72:
#         print(name,len(os.listdir(name_p)))