#  mymodel.py
#  2. Model Architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    """Basic 2-layer Conv3D block: 
    -3D convolutional layers
    -3D batch normalization layers
    -ReLU activation functions"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), # 3D batch normalization
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), # 3D batch normalization
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()

        # encoder
        # 3D max pooling layers
        self.enc1 = ConvBlock3D(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(features[0], features[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock3D(features[1], features[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ConvBlock3D(features[2], features[3])

        # decoder
        # 3D transposed conv (unsampling)
        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(features[3], features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(features[2], features[1])

        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(features[1], features[0])

        # final 1x1x1 conv 
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    # helper to crop skip connection tensors
    def center_crop(self, enc_feature, target_shape):
        _, _, d, h, w = enc_feature.shape
        td, th, tw = target_shape
        d1 = (d - td) // 2
        h1 = (h - th) // 2
        w1 = (w - tw) // 2
        return enc_feature[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck(self.pool3(e3))

        # decoder with cropping for skip connection compatibility
        d3 = self.up3(b)
        e3_cropped = self.center_crop(e3, d3.shape[2:])
        d3 = torch.cat((e3_cropped, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_cropped = self.center_crop(e2, d2.shape[2:])
        d2 = torch.cat((e2_cropped, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_cropped = self.center_crop(e1, d1.shape[2:])
        d1 = torch.cat((e1_cropped, d1), dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

# count parameters when run directly
if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = UNet3D()
    print(f"Total trainable parameters: {count_parameters(model):,}")
