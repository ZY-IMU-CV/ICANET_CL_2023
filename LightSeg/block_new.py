from torch import nn
import torch
from torch.nn import functional as F

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x
        
class inputlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        self.conv=nn.Sequential(nn.Conv2d(3,32,1),
        nn.Conv2d(32,32,3,1,1),
        nn.Conv2d(32,3,1))
        self.bn=norm2d(32)
        self.act=activation()
    def forward(self, x):
        x= self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        x=self.act(x)
        return x
    
class CDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        dilation=dilations[0]

        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=1,groups=out_channels, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x=self.avg(x) 
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        return x
    

class LSeg_context(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage4=CDBlock(3, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock(128, 256, [1],2),
            CDBlock(256,256,[2],1),
            CDBlock(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock(256, 256, [1], 2),
            CDBlock(256,256,[2],1),
            CDBlock(256, 256, [4], 1),
            CDBlock(256, 256, [8], 1)
        )
        self.stage4_down = nn.AvgPool2d(8,8,ceil_mode=True)
        self.stage8_down = nn.AvgPool2d(4,4,ceil_mode=True)
        self.stage16_down = nn.AvgPool2d(2,2,ceil_mode=True)
        
        self.conv1 = ConvBnAct(48+128+256+256,128,1)
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        x =torch.cat((self.stage4_down(x4),self.stage8_down(x8),self.stage16_down(x16),x32))
        x = self.conv1(x)
        return x

class DSConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, stride=stride,bias=False, apply_act=True):
        super(DWConvBnAct, self).__init__()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        self.stride=stride
        self.conv3x3=nn.Conv2d(in_channels,in_channels,3,stride=1,padding=padding,groups=in_channels,stride=stride)
        self.conv1x1=nn.Conv2d(in_channels,out_channels,1)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        if self.stride ==2:
            x= self.avg(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x
        
class LSeg_speticial(nn.Module)
    def __init__(self):
        super().__init__()
        self.conv1 = DSConvBnAct(3,48,2)
        self.conv2 = DSConvBnAct(48,128,2)
        self.conv3 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
    def forward(self,x)
        x =  self.conv1(x)
        x =  self.conv2(x)
        x =  self.conv3(x)
        x =  self.conv4(x)
        return(x)
        
        
class RegSeg(nn.Module)
    def __init__(self,num_classes)
        self.stem = inputlayer()
        self.context = LSeg_context()
        self.speticial = LSeg_speticial()
        self.conv = ConvBnAct(128,64,1)
        self.convclass = nn.Conv2d(128,num_classes,1)
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location='cpu')
            print(type(dic))
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            print(type(dic))
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)
    def forward(self,x)
        input_shape=x.shape[-2:]
        x = self.stem(x)
        x = self.context(x)+F.interpolate(self.specicial(x), size=[128,256], mode='bilinear', align_corners=False)
        x = F.interpolate(x,[256,512], mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.convclass(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
        
    
    
        
    
        


        
        
        
