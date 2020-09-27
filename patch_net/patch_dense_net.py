import torch
#torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.nn.functional as F
import re
import torch.utils.checkpoint as cp
from collections import OrderedDict

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class Conv_Block(nn.Module):
    def __init__(self, din, dout, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
        super(Conv_Block,self).__init__()
        self.bn = nn.BatchNorm2d(din)
        self.act_fn = act_fn
        self.conv = nn.Conv2d(din, dout, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    def forward(self,input):
        return self.conv(self.act_fn(self.bn(input)))
        
class Conv_Block_x2(nn.Module):
    def __init__(self, din, act_fn, dilation=1):
        super(Conv_Block_x2,self).__init__()
        self.conv_0 = Conv_Block(din, din, act_fn, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = Conv_Block(din, din, act_fn, kernel_size=3, padding=dilation, dilation=dilation)
    def forward(self,input):
        x = self.conv_0(input)
        x = self.conv_1(x)
        return x

class Down_Block(nn.Module):
    def __init__(self, din, dout, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
        super(Down_Block,self).__init__()
        self.bn = nn.BatchNorm2d(din)
        self.act_fn = act_fn
        self.conv = nn.Conv2d(din, dout, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self,input):
        return self.avgpool(self.conv(self.act_fn(self.bn(input))))

class Res_Conv_Block(nn.Module):
    def __init__(self, din, act_fn, dilation=1):
        super(Res_Conv_Block,self).__init__()
        self.conv_0 = Conv_Block(din, din, act_fn, kernel_size=1, padding=0, dilation=1)
        self.conv_1 = Conv_Block(din, din, act_fn, kernel_size=3, padding=dilation, dilation=dilation)
    def forward(self,input):
        x = self.conv_0(input)
        x = self.conv_1(x)
        x = x + input
        return x

class Conv_Block_x2(nn.Module):
    def __init__(self, din, act_fn, dilation=1):
        super(Conv_Block_x2,self).__init__()
        self.conv_0 = Conv_Block(din, din, act_fn, kernel_size=1, padding=0, dilation=1)
        self.conv_1 = Conv_Block(din, din, act_fn, kernel_size=3, padding=dilation, dilation=dilation)
    def forward(self,input):
        x = self.conv_0(input)
        x = self.conv_1(x)
        return x
        
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient
    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)
        
class UNet(nn.Module):
    def __init__(self, input_nc, nd_sin, nd_sout):
        super(UNet, self).__init__()

        act_fn = nn.ReLU(inplace=True)
        
        # Encoder
        self.conv_0_a = nn.Conv2d(input_nc+nd_sin, 64, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv_0 = _DenseBlock(4,  64, 4,  16, 0, True)   #  64+4* 16=128
        self.conv_1 = _DenseBlock(4, 128, 4,  32, 0, True)   # 128+4* 32=256
        self.conv_2 = _DenseBlock(4, 256, 4,  64, 0, True)   # 256+4* 64=512
        self.conv_3 = _DenseBlock(4, 512, 4, 128, 0, True)   # 512+4*128=1024
        
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Decoder
        self.conv_4_a = Conv_Block(1024, 512, act_fn, kernel_size=1, padding=0, dilation=1)
        self.conv_4 = Conv_Block(512, 512, act_fn)
        self.conv_5_a = Conv_Block(512, 256, act_fn, kernel_size=1, padding=0, dilation=1)
        self.conv_5 = Conv_Block(256, 256, act_fn)
        self.conv_6_a = Conv_Block(256, 128, act_fn, kernel_size=1, padding=0, dilation=1)
        self.conv_6 = Conv_Block(128, 128, act_fn)
        
        for m in self.modules():        
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
        self.conv_out = nn.Conv2d(128, nd_sout, kernel_size=1, stride=1, padding=0)

    def forward(self, img, sin):
        
        conv_0 = self.conv_0(self.conv_0_a(torch.cat((img, sin), dim=1)))
        conv_1 = self.conv_1(self.avgpool(conv_0))
        conv_2 = self.conv_2(self.avgpool(conv_1))
        conv_3 = self.conv_3(self.avgpool(conv_2))
                
        conv_4 = self.conv_4(self.conv_4_a(F.interpolate(conv_3, scale_factor=2, mode='bilinear')) + conv_2)
        conv_5 = self.conv_5(self.conv_5_a(F.interpolate(conv_4, scale_factor=2, mode='bilinear')) + conv_1)
        conv_6 = self.conv_6(self.conv_6_a(F.interpolate(conv_5, scale_factor=2, mode='bilinear')) + conv_0)
        
        sout = self.conv_out(conv_6)
        
        return sout