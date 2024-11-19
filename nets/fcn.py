
# -*-coding:utf-8 -*-

import torch.nn as nn

    
class Conv_block(nn.Module):
    def  __init__(self, in_channels, out_channels, drop_rate=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.drop_rate=drop_rate
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        if drop_rate:
            self.drop = nn.Dropout(drop_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 =nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)        
        if self.drop_rate:
            x = self.drop(x)
        return x

        

class fcn_Encoder(nn.Module):
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

class fcn_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, basewidth, drop_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upsampling_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                # self.conv_list.append(Conv_block(in_channels=in_channels,out_channels=basewidth*2**(depth-1-i-1),drop_rate=drop_rate))
                self.upsampling_list.append(nn.ConvTranspose2d(in_channels=in_channels,out_channels=basewidth*2**(depth-1-i-1),kernel_size=2,stride=2))
                in_channels = basewidth*2**(depth-i-1-1)
            else:
                self.conv_list.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1))
    def forward(self,x):
        output = x[0]
        for i,up in enumerate(self.upsampling_list):
            # output = conv(output)
            output = up(output)
            # output = torch.concatenate((output,x[i+1]),dim=1)
            output += x[i+1]
        output = self.conv_list[-1](output)
        return output
    
class fcn(nn.Module):
    def __init__(self, in_channels, out_channels, depth, basewidth, drop_rate=0):
        super(fcn, self).__init__()
        self.Encoder = fcn_Encoder(in_channels=in_channels,depth=depth,basewidth=basewidth,drop_rate=drop_rate)
        in_channels = basewidth*2**(depth-1)
        self.Decoder = fcn_Decoder(in_channels=in_channels,out_channels=out_channels,depth=depth,basewidth=basewidth,drop_rate=drop_rate)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = self.activation(x)
        return x