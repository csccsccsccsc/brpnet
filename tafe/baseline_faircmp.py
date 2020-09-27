import torch
#torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from DenseNet import densenet121_backbone
import re

class Conv_Block(nn.Module):
    def __init__(self, din, dout, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
        super(Conv_Block,self).__init__()
        self.bn = nn.BatchNorm2d(din)
        self.act_fn = act_fn
        self.conv = nn.Conv2d(din, dout, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    def forward(self,input):
        return self.conv(self.act_fn(self.bn(input)))

class Conv_Block2(nn.Module):
    def __init__(self, din, dout, act_fn, kernel_size=3, stride=1, padding=1, dilation=1):
        super(Conv_Block2,self).__init__()
        self.bn = nn.BatchNorm2d(din)
        self.conv = nn.Conv2d(din, dout, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    def forward(self,input):
        return self.conv(self.bn(input))

class IAM(nn.Module):
    def __init__(self, din, act_fn, feature_supervision=True):
        super(IAM,self).__init__()
        self.conv_1_0 = nn.Conv2d(din, din, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_1_1 = nn.Conv2d(din, din, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2 = Conv_Block(din*2, din, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_3_0 = Conv_Block(din, din, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_3_1 = Conv_Block(din, din, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.din = din
        self.feature_supervision = feature_supervision
    def forward(self, x0, x1):
        f_x0 = self.conv_1_0(x0)
        f_x1 = self.conv_1_1(x1)
        f_fusion = self.conv_2(torch.cat((f_x0, f_x1), dim=1))
        fout_x0 = self.conv_3_0(f_fusion)
        fout_x1 = self.conv_3_1(f_fusion)
        if self.feature_supervision:
            return fout_x0, fout_x1, f_x0, f_x1
        else:
            return fout_x0, fout_x1

class UNet(nn.Module):
    def __init__(self, input_nc, nd_sout, nd_cout, if_fromscratch=False):
        super(UNet,self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.conv_fdim_trans_3_s = Conv_Block(1024, 256, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_fdim_trans_2_s = Conv_Block(1024, 256, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_1_s = Conv_Block(512, 256, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_0_s = Conv_Block(256, 256, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_3_c = Conv_Block(1024, 256, act_fn, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_fdim_trans_2_c = Conv_Block(1024, 256, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_1_c = Conv_Block(512, 256, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv_fdim_trans_0_c = Conv_Block(256, 256, act_fn, kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.conv_3_s = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_3_c = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)

        self.iam_2 = IAM(256, act_fn)
        self.iam_1 = IAM(256, act_fn)
        self.iam_0 = IAM(256, act_fn)

        for m in self.modules():        
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.out_s_s0 = nn.Conv2d(256, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c_s0 = nn.Conv2d(256, nd_cout, kernel_size=1, stride=1, padding=0)
        self.out_s_s1 = nn.Conv2d(256, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c_s1 = nn.Conv2d(256, nd_cout, kernel_size=1, stride=1, padding=0)
        self.out_s_s2 = nn.Conv2d(256, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c_s2 = nn.Conv2d(256, nd_cout, kernel_size=1, stride=1, padding=0)
        self.out_s_s3 = nn.Conv2d(256, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c_s3 = nn.Conv2d(256, nd_cout, kernel_size=1, stride=1, padding=0)
        
        self.out_s = nn.Conv2d(256, nd_sout, kernel_size=1, stride=1, padding=0)
        self.out_c = nn.Conv2d(256, nd_cout, kernel_size=1, stride=1, padding=0)
        
        self.encoder_backbone = densenet121_backbone(True)

        if not(if_fromscratch):
            state_dict_pretrained = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/densenet121-a639ec97.pth')
            model_dict = self.encoder_backbone.state_dict()
            new_dict = {}
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict_pretrained.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict_pretrained[new_key] = state_dict_pretrained[key]
                    del state_dict_pretrained[key]

            for k, v in state_dict_pretrained.items():
                if k in model_dict:
                    print(k)
                    new_dict.update({k:v})
            model_dict.update(new_dict)
            self.encoder_backbone.load_state_dict(model_dict)
        else:
            for m in self.encoder_backbone.modules():        
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, img):

        # 256, 512, 1024, 1024
        f0, f1, f2, f3 = self.encoder_backbone(img)
        f3_s = self.conv_3_s(self.conv_fdim_trans_3_s(f3))
        f3_c = self.conv_3_c(self.conv_fdim_trans_3_c(f3)) 

        f2_s = self.conv_fdim_trans_2_s(f2) + F.interpolate(f3_s, scale_factor=2, mode='bilinear')
        f2_c = self.conv_fdim_trans_2_c(f2) + F.interpolate(f3_c, scale_factor=2, mode='bilinear')
        f2_s, f2_c, f2_s_s, f2_s_c = self.iam_2(f2_s, f2_c)
        
        f1_s = self.conv_fdim_trans_1_s(f1) + F.interpolate(f2_s, scale_factor=2, mode='bilinear')
        f1_c = self.conv_fdim_trans_1_c(f1) + F.interpolate(f2_c, scale_factor=2, mode='bilinear')
        f1_s, f1_c, f1_s_s, f1_s_c = self.iam_1(f1_s, f1_c)
        
        f0_s = self.conv_fdim_trans_0_s(f0) + F.interpolate(f1_s, scale_factor=2, mode='bilinear')
        f0_c = self.conv_fdim_trans_0_c(f0) + F.interpolate(f1_c, scale_factor=2, mode='bilinear')
        f0_s, f0_c, f0_s_s, f0_s_c= self.iam_0(f0_s, f0_c)


        sout_s3 = self.out_s_s3(f3_s)
        cout_s3 = self.out_c_s3(f3_c)
        
        sout_s2 = self.out_s_s2(f2_s_s)
        cout_s2 = self.out_c_s2(f2_s_c)
        
        sout_s1 = self.out_s_s1(f1_s_s)
        cout_s1 = self.out_c_s1(f1_s_c)
        
        sout_s0 = self.out_s_s0(f0_s_s)
        cout_s0 = self.out_c_s0(f0_s_c)
        
        sout = self.out_s(f0_s)
        cout = self.out_c(f0_c)

        return sout, sout_s0, sout_s1, sout_s2, sout_s3, cout, cout_s0, cout_s1, cout_s2, cout_s3
        
    def get_last_shared_layer(self):
        return f1, f2, f3, f4