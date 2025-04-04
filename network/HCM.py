import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules import *
from .init_weights import init_weights
from .HybridEncoder import he_tiny, Block


def _upsample_like(src,tar):
    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')
    return src

class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2*channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, cnn_feat, mamba_feat):
        combined = torch.cat([cnn_feat, mamba_feat], dim=1)
        gate = self.gate(combined)
        return gate * cnn_feat + (1 - gate) * mamba_feat
    

    
class CFFD(nn.Module):
    # CFFD: Cross-level Feature Fusion Decoder
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
       
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.cnn_conv2=Conv(in_ch*2, in_ch)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.mamba_conv3=Conv(in_ch*2, in_ch)
        
        self.gf1=GatedFusion(in_ch)
        self.fuse_conv1=nn.Sequential(
            Conv(in_ch*2, in_ch),
            Block(in_ch)
        )
        self.gf2=GatedFusion(in_ch)
        self.fuse_conv2=nn.Sequential(
            Conv(in_ch*2, in_ch),
            Block(in_ch)
        )
        self.gf3=GatedFusion(in_ch)
        self.fuse_conv3=nn.Sequential(
            Conv(in_ch*2, in_ch),
            Block(in_ch)
        )
        self.gf4=GatedFusion(in_ch)
        self.fuse_conv4=Block(in_ch)
        


    def forward(self,x1,x2,x3,x4):
        lx1=x1
        lx2=self.cnn_conv2(torch.cat([x2, self.pool1(lx1)], dim=1))
        
        rx4=x4
        rx3=self.mamba_conv3(torch.cat([x3, _upsample_like(rx4, x3)], dim=1))


        fx4=self.gf4( _upsample_like(lx2, rx4), rx4)
        fx4=self.fuse_conv4(fx4)

        fx3= torch.cat([ _upsample_like(fx4, rx3), self.gf3(_upsample_like(lx2, rx3), rx3)], dim=1)
        fx3=self.fuse_conv3(fx3)

        fx2=torch.cat([ _upsample_like(fx3, lx2), self.gf2(lx2, _upsample_like(rx3, lx2))], dim=1)
        fx2=self.fuse_conv2(fx2)

        fx1=torch.cat([ _upsample_like(fx2, lx1), self.gf1(lx1, _upsample_like(rx3, lx1))], dim=1)
        fx1=self.fuse_conv1(fx1)

        return fx1, fx2, fx3, fx4
    
class HCM(nn.Module):
    # HCM: Hybrid CNN-Mamba Network
    
    def __init__(self, 
            backbone="he_tiny", 
            dims=[96, 192, 384, 768], 
            mid_ch=128, 
            num_classes=2,
            **kwargs
        ):
        super().__init__()

        self.encoder=he_tiny()

        out_ch=num_classes
        # Encoder
        self.eside1=Conv(dims[0], mid_ch)
        self.eside2=Conv(dims[1], mid_ch)
        self.eside3=Conv(dims[2], mid_ch)
        self.eside4=Conv(dims[3], mid_ch)

        # Decoder
        self.FF=CFFD(mid_ch,mid_ch)

        self.dside1 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside2 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside3 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside4 = nn.Conv2d(mid_ch,out_ch,3,padding=1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # encoder
        outs = self.encoder(inputs)
        c1, c2, c3, c4 = outs[:4]
    
        c1=self.eside1(c1)
        c2=self.eside2(c2)
        c3=self.eside3(c3)
        c4=self.eside4(c4)

        up1, up2, up3, up4=self.FF(c1,c2,c3,c4)
        
        d1=self.dside1(up1)
        d2=self.dside2(up2)
        d3=self.dside3(up3)
        d4=self.dside4(up4)
      
        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)

        return S1,S2,S3,S4


def HCM_he_tiny(**kwargs):
    model=HCM(backbone="he_tiny", dims=[96, 192, 384, 768], **kwargs)
    return model

