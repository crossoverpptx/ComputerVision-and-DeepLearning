# -- coding: utf-8 --
import torch
import torchvision
from thop import profile

from model import MSCNA

# # Model
# print('==> Building model..')
# model = torchvision.models.alexnet(pretrained=False)
# create model
model = MSCNA(num_classes=3)

dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
