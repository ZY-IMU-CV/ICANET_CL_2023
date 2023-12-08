from torch import nn
import torch
from torch.nn import functional as F

def activation():
    return nn.ReLU(inplace=True)
def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        #self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x
        
class inputlayer_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        self.conv=nn.Conv2d(3,32,3,1,1)
        #self.bn=norm2d(32)
        self.act=activation()
    def forward(self, x):
        x= self.avg(x)
        x = self.conv(x)
        #x = self.bn(x)
        x=self.act(x)
        return x
        
class inputlayer_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        self.conv3x3=nn.Conv2d(3,3,3,stride=1,padding=1,groups=3)
        self.conv1x1=nn.Conv2d(3,32,1)
        self.bn=norm2d(32)
        self.act=activation()
    def forward(self, x):
        x= self.avg(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        x=self.act(x)
        return x
        
class inputlayer_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3=nn.Conv2d(3,3,3,stride=2,padding=1,groups=3)
        self.conv1x1=nn.Conv2d(3,32,1)
        self.bn=norm2d(32)
        self.act=activation()
    def forward(self, x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        x=self.act(x)
        return x
        
class DWConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(DWConvBnAct, self).__init__()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        self.stride=stride
        self.conv3x3=nn.Conv2d(in_channels,in_channels,3,stride=1,padding=padding,groups=in_channels)
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
        
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        dilation=dilations[0]
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=(1,1),groups=out_channels, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
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
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x)
        return x

class CDBlock_0(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        dilation=dilations[0]

        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        #self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=1,groups=out_channels, padding=dilation,dilation=dilation,bias=False)
        #self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        x=self.conv1(x)
        #x=self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x=self.avg(x) 
        x=self.conv2(x)
        #x=self.bn2(x)
        x=self.act2(x)
        return x

class CDBlock_1(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        dilation=dilations[0]

        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=stride,bias=False)
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
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        return x

class CDBlock_2(nn.Module):
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
        if self.stride == 2:
            x=self.avg(x) 
        x=self.conv1(x)
        x=self.bn1(x)
        x = self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        return x

class CDBlock_3(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        dilation=dilations[0]
        self.conv2=nn.Conv2d(in_channels, in_channels,kernel_size=3,stride=1,groups=in_channels, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(in_channels)
        self.act2=activation()
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        if self.stride == 2:
            x=self.avg(x) 
        x=self.conv2(x)
        x=self.bn2(x)
        x = self.act2(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        return x
       
class CDBlock_4(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        dilation=dilations[0]
        self.conv2=nn.Conv2d(in_channels, in_channels,kernel_size=3,stride=1,groups=in_channels, padding=dilation,dilation=dilation,bias=False)
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        if self.stride == 2:
            x=self.avg(x) 
        x=self.conv2(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        return x
'''      
class CDBlock_5(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        dilation=dilations[0]
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=stride,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=1,groups=out_channels, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x = self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        return x    
'''        
class LightSeg_Decoder0(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 =ConvBnAct(128,128,3,1,1,groups=64)
        self.conv8=ConvBnAct(128,64,3,1,1,groups=64)
        self.conv4=DWConvBnAct(72,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class LightSeg_Decoder1(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 64, 1)
        self.head16=ConvBnAct(channels16, 64, 1)
        self.head8=ConvBnAct(channels8, 64, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(64,64,3,1,1,groups=64)
        self.conv8=ConvBnAct(64,64,3,1,1,groups=64)
        self.conv4=DWConvBnAct(72,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class LightSeg_Decoder2(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=DWConvBnAct(72,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class LightSeg_Decoder3(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 64, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1,groups=64)
        self.conv8=ConvBnAct(128,64,3,1,1,groups=64)
        self.conv4=DWConvBnAct(64,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=x8+x4
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class LightSeg_Decoder4(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1,groups=64)
        self.conv8=ConvBnAct(128,64,3,1,1,groups=64)
        self.conv4=DWConvBnAct(72,32,3,1,1)
        self.classifier=nn.Conv2d(32, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class LightSeg_Decoder5(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 =ConvBnAct(128,128,3,1,1,groups=32)
        self.conv8=ConvBnAct(128,64,3,1,1,groups=32)
        self.conv4=DWConvBnAct(72,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
class LightSeg_Decoder6(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 =ConvBnAct(128,128,3,1,1,groups=64)
        self.conv8=ConvBnAct(128,64,3,1,1,groups=32)
        self.conv4=DWConvBnAct(72,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
class LightSeg_Decoder7(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 =ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8=ConvBnAct(128,64,3,1,1,groups=32)
        self.conv4=DWConvBnAct(72,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
class LightSeg_Decoder8(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 =ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8=ConvBnAct(128,64,3,1,1,groups=64)
        self.conv4=DWConvBnAct(72,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class LightSeg_Encoder_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_0(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_H_Encoder_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=DBlock(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1],2),
            DBlock(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1],2),
            DBlock(256,256,[2],1),
            DBlock(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], 2),
            DBlock(256,256,[2],1),
            DBlock(256, 256, [4], 1),
            DBlock(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_R18_Encoder_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=DBlock(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1],2),
            DBlock(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1],2),
            DBlock(256,256,[1],1),
            DBlock(256, 256, [1],1),
            DBlock(256, 256, [1],1),
            DBlock(256,256,[2],1),
            DBlock(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], 2),
            DBlock(256,256,[1],1),
            DBlock(256, 256, [1], 1),
            DBlock(256, 256, [1], 1),
            DBlock(256, 256, [1], 1),
            DBlock(256,256,[2],1),
            DBlock(256, 256, [4], 1),
            DBlock(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_Encoder_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_1()
        self.stage4=CDBlock_0(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_Encoder_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_2()
        self.stage4=CDBlock_0(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_Encoder_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBnAct(3,32,3,2,1)
        self.stage4=CDBlock_0(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_Encoder_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_1(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_1(48, 128, [1],2),
            CDBlock_1(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_1(128, 256, [1],2),
            CDBlock_1(256,256,[2],1),
            CDBlock_1(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_1(256, 256, [1], 2),
            CDBlock_1(256,256,[2],1),
            CDBlock_1(256, 256, [4], 1),
            CDBlock_1(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_Encoder_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_2(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_2(48, 128, [1],2),
            CDBlock_2(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_2(128, 256, [1],2),
            CDBlock_2(256,256,[2],1),
            CDBlock_2(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_2(256, 256, [1], 2),
            CDBlock_2(256,256,[2],1),
            CDBlock_2(256, 256, [4], 1),
            CDBlock_2(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_Encoder_6(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_3(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_3(48, 128, [1],2),
            CDBlock_3(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_3(128, 256, [1],2),
            CDBlock_3(256,256,[2],1),
            CDBlock_3(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_3(256, 256, [1], 2),
            CDBlock_3(256,256,[2],1),
            CDBlock_3(256, 256, [4], 1),
            CDBlock_3(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class LightSeg_Encoder_7(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_4(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_4(48, 128, [1],2),
            CDBlock_4(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_4(128, 256, [1],2),
            CDBlock_4(256,256,[2],1),
            CDBlock_4(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_4(256, 256, [1], 2),
            CDBlock_4(256,256,[2],1),
            CDBlock_4(256, 256, [4], 1),
            CDBlock_4(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
'''   
class LightSeg_Encoder_8(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_5(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_5(48, 128, [1],2),
            CDBlock_5(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_5(128, 256, [1],2),
            CDBlock_5(256,256,[2],1),
            CDBlock_5(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_5(256, 256, [1], 2),
            CDBlock_5(256,256,[2],1),
            CDBlock_5(256, 256, [4], 1),
            CDBlock_5(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
'''      
class LightSeg_Encoder_9(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_0(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[3],1),
            CDBlock_0(256, 256, [6],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 2),
            CDBlock_0(256,256,[3],1),
            CDBlock_0(256, 256, [6], 1),
            CDBlock_0(256, 256, [9], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class LightSeg_Encoder_10(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=nn.Sequential(
            CDBlock_0(32, 48, [1], 2),
            CDBlock_0(48, 48, [1], 1))
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 2),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class LightSeg_Encoder_11(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=nn.Sequential(
            CDBlock_0(32, 48, [1], 2),
            CDBlock_0(48, 48, [1], 1))
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_Encoder_12(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=nn.Sequential(
            CDBlock_0(32, 48, [1], 2),
            CDBlock_0(48, 48, [1], 1))
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],1),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class LightSeg_Encoder_13(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_0(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 1),
            CDBlock_0(256, 256, [2], 1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
    
class LightSeg_Encoder_14(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_0(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],1),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 1),
            CDBlock_0(256, 256, [2], 1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
class LightSeg_Encoder_RS18(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=DBlock(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1],2),
            DBlock(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1],2),
            DBlock(256,256,[2],1),
            DBlock(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], 2),
            DBlock(256,256,[2],1),
            DBlock(256, 256, [4], 1),
            DBlock(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3,3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet_18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,32,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(ResBlock(32,48,2),ResBlock(48,48,1))
        self.conv3 = nn.Sequential(ResBlock(48,128,2),ResBlock(128,128,1))
        self.conv4 = nn.Sequential(ResBlock(128,256,2),ResBlock(256,256,1))
        self.conv5 = nn.Sequential(ResBlock(256, 256, 2), ResBlock(256, 256, 1))
    def forward(self,x):
        x = self.conv1(x)
        x4 = self.conv2(x)
        x8 = self.conv3(x4)
        x16 = self.conv4(x8)
        x32 = self.conv5(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}
        