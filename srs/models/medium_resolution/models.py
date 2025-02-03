import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

in_channels_variable =  9
out_channels_autoencoder = 9

# Convolutional block to be used in autoencoder and U-Net
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.15):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x

# Encoder block: Conv block followed by maxpooling
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.15):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p

# Encoder will be the same for Autoencoder and U-net
class Encoder(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(Encoder, self).__init__()
        self.enc1 = EncoderBlock(in_channels_variable, 64, dropout_prob)
        self.enc2 = EncoderBlock(64, 128, dropout_prob)
        self.enc3 = EncoderBlock(128, 256, dropout_prob)
        self.enc4 = EncoderBlock(256, 512, dropout_prob)
        self.bridge = ConvBlock(512, 1024, dropout_prob)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        encoded = self.bridge(p4)
        return encoded, (s1, s2, s3, s4)
    
# Decoder block for autoencoder (no skip connections)
class DecoderBlockForAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.15):
        super(DecoderBlockForAutoencoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels, out_channels, dropout_prob)

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv_block(x)
        return x

# Decoder for Autoencoder ONLY
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec1 = DecoderBlockForAutoencoder(1024, 512)
        self.dec2 = DecoderBlockForAutoencoder(512, 256)
        self.dec3 = DecoderBlockForAutoencoder(256, 128)
        self.dec4 = DecoderBlockForAutoencoder(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels_autoencoder, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

# Use encoder and decoder blocks to build the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def summary(self, input_size):
        summary(self, input_size)

# Decoder block for unet
class DecoderBlockForUnet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.15):
        super(DecoderBlockForUnet, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels, dropout_prob)

    def forward(self, x, skip_features):
        x = self.upconv(x)
        x = torch.cat((x, skip_features), dim=1)
        x = self.conv_block(x)
        return x
    
# Build Unet using the blocks
class Unet(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.25):
        super(Unet, self).__init__()
        self.encoder = Encoder(dropout_prob)
        self.bridge = ConvBlock(512, 1024, dropout_prob)
        self.dec1 = DecoderBlockForUnet(1024, 512, dropout_prob)
        self.dec2 = DecoderBlockForUnet(512, 256, dropout_prob)
        self.dec3 = DecoderBlockForUnet(256, 128, dropout_prob)
        self.dec4 = DecoderBlockForUnet(128, 64, dropout_prob)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.activation = nn.Identity()

    def forward(self, x):
        encoded, (s1, s2, s3, s4) = self.encoder(x)
        x = self.dec1(encoded, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)
        x = self.final_conv(x)
        x = self.activation(x)
        return x

    def summary(self, input_size):
        summary(self, input_size)

# Example usage:
# autoencoder = Autoencoder()
# print(autoencoder)

# unet = Unet()
# print(unet)