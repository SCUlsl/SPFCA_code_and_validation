
# -*-coding:utf-8 -*-

import torch
import torch.nn as nn
    
class Conv_block(nn.Module):
    def  __init__(self, in_channels, out_channels, drop_rate=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drop_rate=drop_rate
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.normal1 = nn.GroupNorm(out_channels,out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        if drop_rate:
            self.drop = nn.Dropout(drop_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.normal2 = nn.GroupNorm(out_channels,out_channels)
        self.relu2 =nn.ReLU(inplace=True)
        if in_channels!=out_channels:
            self.resample = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
        else:
            self.resample = False
    def forward(self,x):
        identify = x
        x = self.conv1(x)
        x = self.normal1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.normal2(x)
        if self.resample:
            identify = self.resample(identify)
        x += identify
        x = self.relu2(x)        
        if self.drop_rate:
            x = self.drop(x)
        return x

  
class resunet_Encoder(nn.Module):
    def __init__(self, in_channels, depth, basewidth, drop_rate,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drop_rate = drop_rate
        self.conv_list = nn.ModuleList()
        self.downsampling = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.conv_list.append(Conv_block(in_channels=in_channels,out_channels=basewidth*(2**i),drop_rate=drop_rate))
                self.downsampling.append(nn.AvgPool2d(kernel_size=2,stride=2))
                in_channels = basewidth*(2**i)
            else:
                self.conv_list.append(Conv_block(in_channels=in_channels,out_channels=basewidth*(2**i),drop_rate=drop_rate))

    def forward(self,x):
        output_list = []
        for conv,down in zip(self.conv_list[:-1],self.downsampling):
            x = conv(x)
            output_list.insert(0,x)
            x = down(x)
        x = self.conv_list[-1](x)
        output_list.insert(0,x)
        return output_list

class resunet_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, basewidth, drop_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upsampling_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.conv_list.append(Conv_block(in_channels=in_channels,out_channels=basewidth*2**(depth-1-i-1),drop_rate=drop_rate))
                self.upsampling_list.append(nn.UpsamplingBilinear2d(scale_factor=2))
                in_channels = basewidth*2**(depth-i-1)
            else:
                self.conv_list.append(Conv_block(in_channels=in_channels,out_channels=out_channels,drop_rate=drop_rate))
    def forward(self,x):
        output = x[0]
        for i,(conv,up) in enumerate(zip(self.conv_list[:-1],self.upsampling_list)):
            output = conv(output)
            output = up(output)
            output = torch.concatenate((output,x[i+1]),dim=1)
        output = self.conv_list[-1](output)
        return output
    
class resunet(nn.Module):
    def __init__(self, in_channels, out_channels, depth, basewidth, drop_rate=0):
        super(resunet, self).__init__()
        self.Encoder = resunet_Encoder(in_channels=in_channels,depth=depth,basewidth=basewidth,drop_rate=drop_rate)
        in_channels = basewidth*2**(depth-1)
        self.Decoder = resunet_Decoder(in_channels=in_channels,out_channels=out_channels,depth=depth,basewidth=basewidth,drop_rate=drop_rate)
        self.final = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = self.final(x)
        x = self.activation(x)
        return x


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)