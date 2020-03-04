from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, num_classes, args):
        super(Net, self).__init__()
        # Model Selection
        original_model = models.__dict__['vgg16'](pretrained=True)


        self.features = original_model.features
        if num_classes is not None:
            self.classifier = nn.Sequential(OrderedDict([
                ('do1', nn.Dropout()),
                ('fc1', nn.Linear(25088, 4096)),
                ('fc_relu1', nn.ReLU(inplace=True)),
                ('do2', nn.Dropout()),
                ('fc2', nn.Linear(4096, 4096)),
                ('fc_relu2', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes))
            ]))
        else:
            self.classifier = original_model.classifier
            self.softmax = nn.Softmax(dim=1)



    def forward(self, x):
        in_size = x.size(0)
        x = self.features(x)
        # x = x.view(in_size, 256 * 6 * 6) #alexnet
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.classifier(x)

        return x
#
# class Net(nn.Module):
#     def __init__(self, num_classes, args):
#         super(Net, self).__init__()
#
#         self.features = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)),
#             ('elu1', nn.ELU(inplace=True)),
#             ('batch1', nn.BatchNorm2d(64, eps=0.001)),
#
#             ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
#             ('elu2', nn.ELU(inplace=True)),
#             ('batch2', nn.BatchNorm2d(64, eps=0.001)),
#
#             ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
#             ('elu3', nn.ELU(inplace=True)),
#             ('batch3', nn.BatchNorm2d(128, eps=0.001)),
#
#             ('conv3', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
#             ('elu3', nn.ELU(inplace=True)),
#             ('batch3', nn.BatchNorm2d(128, eps=0.001)),
#
#             ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
#
#             ('conv3', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
#             ('elu3', nn.ELU(inplace=True)),
#             ('batch3', nn.BatchNorm2d(256, eps=0.001)),
#
#             ('conv3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
#             ('elu3', nn.ELU(inplace=True)),
#             ('batch3', nn.BatchNorm2d(256, eps=0.001)),
#
#             ('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
#         ]))
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(16*16*256, 2048),
#             nn.ELU(inplace=True),
#             nn.BatchNorm2d(2048, eps=0.001),
#             nn.Linear(2048, num_classes)
#         )
#
#         nn.init.xavier_uniform_(self.classifier[1].weight)
#         nn.init.xavier_uniform_(self.classifier[4].weight)
#         self.modelName = 'ECG_CNN'
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 16 * 16 * 256) #flatten the tensor
#         x = self.classifier(x)
#
#         return x

if __name__ == '__main__':
    # Dummy arguments
    class args:
        def __init__(self):
            self.nclass = 10

    n = Net(10, args())
    print(n(torch.zeros(20, 1, 64, 64)).shape)