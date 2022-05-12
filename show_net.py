import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import netron
from tensorboardX import SummaryWriter
from torch.autograd import Variable

# from lib.models.fpn import LResNet50E_IR_Occ as LResNet50E_IR_FPN

from model import FaceMobileNet, ResIRSE

# model = FaceMobileNet(512)  # 模型
model = ResIRSE(512,0.5)  # 模型
data = torch.rand(1, 1, 128, 128)  # 数据
onnx_path = "ResIRSE.onnx"  # 文件名
# torch.onnx.export(model, …)
torch.onnx.export(model, data, onnx_path, opset_version=11)  # 导出神经网络模型为onnx格式
netron.start(onnx_path)  # 启动netron


