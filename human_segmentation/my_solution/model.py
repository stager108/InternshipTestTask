import torch

import torch.nn as nn

def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
    net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))
    return net;


class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        
        n = 4
        super(SegmenterModel, self).__init__()
        self.conv1 = conv_bn_relu(in_size, 8*n)
        
        self.conv3 = conv_bn_relu(8*n, 16*n)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = conv_bn_relu(16*n, 32*n)        
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = conv_bn_relu(32*n, 64*n)
        
        self.conv8 = conv_bn_relu(64*n, 32*n)
  
        self.conv9 = conv_bn_relu(64*n, 32*n)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv10 = conv_bn_relu(32*n, 16*n)
        
        self.conv11 = conv_bn_relu(32*n, 16*n)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv12 = conv_bn_relu(16*n, 8*n)
        
        self.conv13 = conv_bn_relu(16*n, 8*n)
        self.conv14 =  nn.Conv2d(8*n, 1, kernel_size=1)      
         
    def forward(self, input):
        x = self.conv1(input)
        x_saved1 = x
        #print(x.size())
        x = self.mp2(self.conv3(x))
        x_saved2 = x
        #print(x.size())
        x = self.mp3(self.conv5(x))
        x_saved3 = x
        #print(x.size())
        x = self.conv8(self.conv7(x))
        x = torch.cat((x_saved3, x), dim=1)
        x = self.conv10(self.upsample1(self.conv9(x)))
        #print(x.size())
        x = torch.cat((x_saved2, x), dim=1)
        x = self.conv12(self.upsample2(self.conv11(x)))
        #print(x.size())
        x = torch.cat((x_saved1, x), dim=1)
        x = self.conv14(self.conv13(x))
        #print(x.size())
        return x.clamp(0,1)

