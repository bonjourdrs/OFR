import torch
import torchvision.transforms as T

class Config:
    #logdir
    OUTPUT_DIR = 'TensorBoard'
    # network settings
    # backbone = 'ARC-M' # [ARC-R, ARC-M, FMask-R, FMask-M,FaceNet]
    # backbone = 'ARC-R' # [ARC-R, ARC-M, FMask-R, FMask-M,FaceNet]
    # backbone = 'FMask-R' # [ARC-R, ARC-M, FMask-R, FMask-M,FaceNet]
    # backbone = 'FMask-M' # [ARC-R, ARC-M, FMask-R, FMask-M,FaceNet]
    backbone = 'FMask-M_L' # [ARC-R, ARC-M, FMask-R, FMask-M,FaceNet]
    # backbone = 'FMask-M_qua' # [ARC-R, ARC-M, FMask-R, FMask-M,FaceNet]

    # if backbone=='ARC-R' or backbone=='ARC-M':
    #     net_type=0
    # elif backbone=='FMask-R' or backbone=='FMask-M' or backbone=='FMask-M_L' or backbone=='FMask-M_qua':
    #     net_type=1
    # else:
    #     net_type=2

    if backbone == 'ARC-R':
        net_type=0
        # test_model = './pretrained/ARC-R2.pth.tar'
        test_model = './pretrained/ARC-R_best.pth.tar'
    elif backbone == 'ARC-M':
        net_type=0
        # test_model = "./pretrained/ARC-M.pth.tar"
        # test_model = "./pretrained/ARC-M2.pth.tar"
        test_model = "./pretrained/ARC-M_best.pth.tar"
    elif backbone == 'FMask-R':
        net_type=1
        # test_model = "./pretrained/c11_best_p5.pth.tar"
        # test_model = "./pretrained/c11_p5.pth.tar"
        # test_model = "./pretrained/c12_best_p5.pth.tar"
        # test_model = "./pretrained/c12_p5.pth.tar"
        # test_model = "./pretrained/model_p4_baseline_9938_8205_3610.pth.tar"
        test_model = './pretrained/model_p5_w1_9938_9470_6503.pth.tar'
    elif backbone == 'FMask-M':
        net_type=1
        # test_model = "./pretrained/mobilenet_p5.pth.tar"
        test_model = "./pretrained/mobilenet_p5_best.pth.tar"
    elif backbone == 'FMask-M_L':
        net_type=1
        # test_model = './pretrained/Fmask_M_L.pth.tar'
        test_model = './pretrained/Fmask_M_L_best.pth.tar'
    elif backbone == 'FMask-M_qua':
        net_type=1
        test_model = './pretrained/int8_best_qua.pth.tar'
        # test_model = './pretrained/int8_qua.pth.tar'
        # test_model = './pretrained/int4_qua.pth.tar'
        # test_model = './pretrained/int4_best_qua.pth.tar'
    elif backbone == 'FaceNet':
        net_type=2
        test_model = './pretrained/backbone_weights_of_inception_resnetv1.pth'

    LFW_PATH = 'data/datasets/lfw-occ/lfw-112X96/'
    LFW_OCC_PATH = 'data/datasets/lfw-occ/lfw-112X96_occ/'
    # LFW_OCC_PATH = 'data/datasets/lfw-occ/lfw-112X96_glasses2/'
    # LFW_OCC_PATH = 'data/datasets/lfw-occ/lfw-112X96_masked2/'
    LFW_PAIRS = 'data/datasets/lfw-occ/lfw_112x96_occ_pairs.txt'


    # test_model = "./pretrained/fm_109.pth"
    # test_model = "./pretrained/train2/train2_res_42.pth"
    # test_model = "./pretrained/train2/train2_fm_127.pth"
    # test_model = "./pretrained/res_23.pth"


    metric = 'arcface'  # [cosface, arcface]
    embedding_size = 512
    drop_ratio = 0.5
    if  test_model == "./pretrained/model_p4_baseline_9938_8205_3610.pth.tar":
        S=4
    else:
        S = 5
    NUM_MASK = int(((S + 1) * S / 2) ** 2) + 1
    #LOSS
    WEIGHT_PRED = 1.0

    # config.DATASET.LMDB_FILE = 'data/datasets/CASIA-112x96-LMDB.lmdb'
    PRINT_FREQ=100
    TRAIN_DATASET='Webface_occ'
    LMDB_FILE = 'data/datasets/Webface_112x96_S'+str(S)+'.lmdb'




    # training settings
    checkpoints = "pretrained"

    restore = False
    restore_model = "train2_fm_104.pth"
    
    train_batch_size = 64
    test_batch_size = 64
    SHUFFLE = True
    Test_SHUFFLE = False
    epoch = 40
    optimizer = 'sgd'  # ['sgd', 'adam']
    if backbone == 'FMask-M_qua':
        lr = 1e-3
        lr_step = 10
    else:
        lr = 1e-2
        lr_step = 15
    MOMENTUM = 0.9
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss' # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pin_memory = True  # if memory is large, set it True to speed up a bit
    num_workers = 4  # dataloader

    # data preprocess
    input_shape = [3, 112, 96]
    train_transform = T.Compose([
        # T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((140, 120)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    test_transform = T.Compose([
        # T.Grayscale(),
        # T.Pad(22,0),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

config = Config()
