"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import math
from functools import partial
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nonlinearity = partial(F.relu,inplace=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Dblock_more_dilate(nn.Module):
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out


import torch.fft as fft  # 用于傅里叶变换
class FrequencyEnhancement(nn.Module):
    def __init__(self, embed_dim):
        super(FrequencyEnhancement, self).__init__()
        # 卷积层用于对频率信息进行变换
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # 进行2D傅里叶变换，提取频率信息
        x_freq = fft.fft2(x, dim=(-2, -1))  # 对最后两个维度(H, W)进行2D傅里叶变换
        x_freq = torch.abs(x_freq)  # 取幅值
        x_freq = self.conv(x_freq)  # 对频域信息进行卷积处理

        return x_freq

class WaveFormer(nn.Module):
    def __init__(self, embed_dim):
        super(WaveFormer, self).__init__()
        # 小波变换的低通和高通滤波器
        self.low_pass = nn.Parameter(torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]), requires_grad=False)
        self.high_pass = nn.Parameter(torch.tensor([[[[0.5, -0.5], [0.5, -0.5]]]]), requires_grad=False)

        # 卷积层用于对分解后的频率信息进行变换
        self.conv_low = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.conv_high = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # 对输入进行小波变换分解
        low_freq, high_freq = self.dwt(x)

        # 对低频和高频成分分别进行卷积处理
        low_freq = self.conv_low(low_freq)
        high_freq = self.conv_high(high_freq)

        # 确保低频和高频特征图的分辨率是原始特征图的一半
        low_freq = F.interpolate(low_freq, size=(x.size(-2), x.size(-1)), mode='nearest')
        high_freq = F.interpolate(high_freq, size=(x.size(-2), x.size(-1)), mode='nearest')

        return low_freq, high_freq

    def dwt(self, x):
        # 对输入进行二维小波变换分解
        # 使用低通和高通滤波器
        b, c, h, w = x.shape

        # 扩展输入以适应滤波器
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # 低频成分
        low = F.conv2d(x, self.low_pass.repeat(c, 1, 1, 1), groups=c)
        low = low[:, :, ::2, ::2]

        # 高频成分
        high = F.conv2d(x, self.high_pass.repeat(c, 1, 1, 1), groups=c)
        high = high[:, :, ::2, ::2]

        return low, high
class HorizontalTransformer1(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(HorizontalTransformer1, self).__init__()
        self.layers1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.layers2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.freq_enhancement = WaveFormer(embed_dim)
        self.fusion_conv = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)  # 用于融合的卷积层

    def forward(self, x):
        B, C, H, W = x.size()

        # 提取频率增强特征
        low_frequency, high_frequency = self.freq_enhancement(x)

        # 对输入进行维度变换，适应Transformer操作
        low_frequency = low_frequency.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        for layer in self.layers1:
            low_frequency = layer(low_frequency)
        low_frequency = low_frequency.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 对输入进行维度变换，适应Transformer操作
        high_frequency = high_frequency.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        for layer in self.layers2:
            high_frequency = layer(high_frequency)
        high_frequency = high_frequency.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 将频域特征与Transformer输出融合
        fused_features = torch.cat([x, low_frequency, high_frequency], dim=1)
        # 通过卷积层融合拼接后的特征
        x = self.fusion_conv(fused_features)
        return x


class HorizontalTransformer2(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(HorizontalTransformer2, self).__init__()
        self.layers1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.layers2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.freq_enhancement = WaveFormer(embed_dim)
        self.fusion_conv = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)  # 用于融合的卷积层

    def forward(self, x):
        B, C, H, W = x.size()

        # 提取频率增强特征
        low_frequency, high_frequency = self.freq_enhancement(x)

        # 对输入进行维度变换，适应Transformer操作
        low_frequency = low_frequency.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        for layer in self.layers1:
            low_frequency = layer(low_frequency)
        low_frequency = low_frequency.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 对输入进行维度变换，适应Transformer操作
        high_frequency = high_frequency.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        for layer in self.layers2:
            high_frequency = layer(high_frequency)
        high_frequency = high_frequency.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 将频域特征与Transformer输出融合
        fused_features = torch.cat([ low_frequency, high_frequency], dim=1)
        # 通过卷积层融合拼接后的特征
        x = self.fusion_conv(fused_features)
        return x



class VerticalTransformer1(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(VerticalTransformer1, self).__init__()
        self.layers1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.layers2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.freq_enhancement = WaveFormer(embed_dim)
        self.fusion_conv = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)  # 用于融合的卷积层

    def forward(self, x):
        B, C, H, W = x.size()

        # 提取频率增强特征
        low_frequency, high_frequency = self.freq_enhancement(x)

        # 对输入进行维度变换，适应Transformer操作
        low_frequency = low_frequency.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        for layer in self.layers1:
            low_frequency = layer(low_frequency)
        low_frequency = low_frequency.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        # 对输入进行维度变换，适应Transformer操作
        high_frequency = high_frequency.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        for layer in self.layers2:
            high_frequency = layer(high_frequency)
        high_frequency = high_frequency.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        # 将频域特征与Transformer输出融合
        fused_features = torch.cat([x, low_frequency, high_frequency], dim=1)
        # 通过卷积层融合拼接后的特征
        x = self.fusion_conv(fused_features)
        return x


class VerticalTransformer2(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(VerticalTransformer2, self).__init__()
        self.layers1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.layers2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.freq_enhancement = WaveFormer(embed_dim)
        self.fusion_conv = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)  # 用于融合的卷积层

    def forward(self, x):
        B, C, H, W = x.size()

        # 提取频率增强特征
        low_frequency, high_frequency = self.freq_enhancement(x)

        # 对输入进行维度变换，适应Transformer操作
        low_frequency = low_frequency.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        for layer in self.layers1:
            low_frequency = layer(low_frequency)
        low_frequency = low_frequency.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        # 对输入进行维度变换，适应Transformer操作
        high_frequency = high_frequency.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        for layer in self.layers2:
            high_frequency = layer(high_frequency)
        high_frequency = high_frequency.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        # 将频域特征与Transformer输出融合
        fused_features = torch.cat([ low_frequency, high_frequency], dim=1)
        # 通过卷积层融合拼接后的特征
        x = self.fusion_conv(fused_features)
        return x

class AxialContext(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AxialContext, self).__init__()
        self.HTransformer = HorizontalTransformer1(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*2,  num_layers=1)
        self.VTransformer = VerticalTransformer1(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim, num_layers=1)
        self.fusion_conv = nn.Conv2d(3 * embed_dim, embed_dim, kernel_size=1)

    def forward(self,x):

        Hcontext = self.HTransformer(x)
        Vcontext = self.VTransformer(x)
        fused_out = torch.cat([Hcontext, Vcontext, x], dim=1)
        fused_out = self.fusion_conv(fused_out)
        return fused_out

class AxialContext1(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AxialContext1, self).__init__()
        self.HTransformer = HorizontalTransformer2(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*2,  num_layers=1)
        self.VTransformer = VerticalTransformer2(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim, num_layers=1)
        self.fusion_conv = nn.Conv2d(3 * embed_dim, embed_dim, kernel_size=1)

    def forward(self,x):

        Hcontext = self.HTransformer(x)
        Vcontext = self.VTransformer(x)
        fused_out = torch.cat([Hcontext, Vcontext, x], dim=1)
        fused_out = self.fusion_conv(fused_out)
        return fused_out
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class GaussianUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super(GaussianUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # 初始化高斯核参数
        self.sigma_x = nn.Parameter(torch.ones(1))  # 可学习的 x 方向标准差
        self.sigma_y = nn.Parameter(torch.ones(1))  # 可学习的 y 方向标准差
        self.opacity = nn.Parameter(torch.ones(1))  # 可学习的透明度

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # 生成高斯核
        kernel = self._generate_gaussian_kernel()
        kernel = kernel.to(x.device)  # 确保在同一个设备上

        # 上采样
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

        # 逐通道应用高斯核
        gaussian_out = []
        for i in range(channels):
            channel_out = F.conv2d(upsampled[:, i:i + 1, :, :], kernel, padding=self.kernel_size // 2)
            gaussian_out.append(channel_out)

        # 将所有通道的结果拼接回一个张量
        gaussian_out = torch.cat(gaussian_out, dim=1)

        # 应用卷积
        out = self.conv(gaussian_out)
        return out

    def _generate_gaussian_kernel(self):
        """生成高斯核"""
        # 确保所有张量都在与输入张量相同的设备上
        device = self.sigma_x.device  # 获取参数所在的设备

        ax = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, device=device).float()
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx ** 2 / (2 * self.sigma_x ** 2) + (yy ** 2 / (2 * self.sigma_y ** 2))))
        kernel = kernel / (2 * torch.pi * self.sigma_x * self.sigma_y)  # 归一化
        kernel = kernel * self.opacity  # 应用透明度
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)  # 调整形状
        return kernel

class FHNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(FHNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.context =  AxialContext(embed_dim=512, num_heads=8)


        self.decoder4 = DecoderBlock(filters[3] // 2, filters[2] // 2)
        self.gaussian_upsample4 = GaussianUpsample(filters[3] // 2, filters[2] // 2, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2] // 2, filters[1] // 2)
        self.gaussian_upsample3 = GaussianUpsample(filters[2] // 2, filters[1] // 2, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1] // 2, filters[0] // 2)
        self.gaussian_upsample2 = GaussianUpsample(filters[1] // 2, filters[0] // 2, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0] // 2, filters[0] // 2)
        self.gaussian_upsample1 = GaussianUpsample(filters[0] // 2, filters[0] // 2, kernel_size=3, scale_factor=2)
        self.conv1x1_4 = nn.Conv2d(filters[2], filters[2], kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(filters[1], filters[1], kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(filters[0], filters[0], kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(filters[0], filters[0], kernel_size=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection,conv1x1):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)
        x_gaussian = x[:, :C // 2, :, :]  # 高斯核部分
        x_decoder = x[:, C // 2:, :, :]  # 普通解码器部分

        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return conv1x1(out + skip_connection)  # 加上跳跃连接


    def forward(self, image):

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.context(e4)

        # Decoder
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3,self.conv1x1_4)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2,self.conv1x1_3)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1,self.conv1x1_2)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x,self.conv1x1_1)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return F.sigmoid(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_channels, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_channels, embed_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.output_linear = nn.Linear(embed_dim, input_channels//2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # Flatten the spatial dimensions and embed the channels
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (N, B, C)
        x = self.embedding(x)  # (N, B, D)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.output_linear(x)  # (N, B, 256)

        # Reshape back to (B, 256, 64, 64)

        x = x.permute(1, 2, 0).view(b, c//2, h, w)
        x = self.up(x)

        return x
class FourierFormer(nn.Module):
    def __init__(self, embed_dim):
        super(FourierFormer, self).__init__()
        self.embed_dim = embed_dim
        # 可学习的高低频掩码参数（增强特征适配性）
        self.low_freq_mask = nn.Parameter(torch.ones(1, embed_dim, 1, 1))
        self.high_freq_mask = nn.Parameter(torch.ones(1, embed_dim, 1, 1))

    def forward(self, x):
        """
        傅里叶变换实现高低频分离
        输入: x (B, C, H, W) - 道路特征图
        输出: low_frequency (低频特征), high_frequency (高频特征)
        """
        B, C, H, W = x.size()

        # 1. 二维傅里叶变换（复数域）
        fft = torch.fft.fft2(x, dim=(-2, -1))  # (B, C, H, W) 复数张量
        fft_shift = torch.fft.fftshift(fft)  # 低频移至中心

        # 2. 构建高低频掩码（中心区域为低频，边缘为高频）
        crow, ccol = H // 2, W // 2  # 中心坐标
        mask_radius = min(crow, ccol) // 4  # 低频掩码半径（可调整）

        # 初始化掩码
        low_mask = torch.zeros_like(x)
        high_mask = torch.ones_like(x)

        # 中心区域设为1（低频），其余为0；高频掩码相反
        low_mask[:, :, crow - mask_radius:crow + mask_radius, ccol - mask_radius:ccol + mask_radius] = 1
        high_mask = 1 - low_mask

        # 3. 分离高低频并逆变换
        # 低频分量
        fft_low = fft_shift * low_mask * self.low_freq_mask
        low_frequency = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_low), dim=(-2, -1)))

        # 高频分量（边缘/细节）
        fft_high = fft_shift * high_mask * self.high_freq_mask
        high_frequency = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_high), dim=(-2, -1)))

        # 特征投影（匹配原WaveFormer输出维度）
        low_frequency = nn.functional.conv2d(low_frequency, nn.Conv2d(C, C, 1).weight.to(x.device), bias=None)
        high_frequency = nn.functional.conv2d(high_frequency, nn.Conv2d(C, C, 1).weight.to(x.device), bias=None)

        return low_frequency, high_frequency


# 水平Transformer（傅里叶版）
class HorizontalTransformerfft(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(HorizontalTransformerfft, self).__init__()
        self.layers1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.layers2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.freq_enhancement = FourierFormer(embed_dim)  # 替换为傅里叶变换
        self.fusion_conv = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        # 傅里叶变换提取高低频
        low_frequency, high_frequency = self.freq_enhancement(x)

        # 水平Transformer处理低频
        low_frequency = low_frequency.permute(0, 2, 3, 1).reshape(B * H, W, C)
        for layer in self.layers1:
            low_frequency = layer(low_frequency)
        low_frequency = low_frequency.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 水平Transformer处理高频
        high_frequency = high_frequency.permute(0, 2, 3, 1).reshape(B * H, W, C)
        for layer in self.layers2:
            high_frequency = layer(high_frequency)
        high_frequency = high_frequency.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 融合原始特征+高低频特征
        fused_features = torch.cat([x, low_frequency, high_frequency], dim=1)
        x = self.fusion_conv(fused_features)
        return x

# 垂直Transformer（傅里叶版）
class VerticalTransformerfft(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(VerticalTransformerfft, self).__init__()
        self.layers1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.layers2 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.freq_enhancement = FourierFormer(embed_dim)  # 替换为傅里叶变换
        self.fusion_conv = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        low_frequency, high_frequency = self.freq_enhancement(x)

        # 垂直Transformer处理低频
        low_frequency = low_frequency.permute(0, 3, 2, 1).reshape(B * W, H, C)
        for layer in self.layers1:
            low_frequency = layer(low_frequency)
        low_frequency = low_frequency.reshape(B, W, H, C).permute(0, 3, 2, 1)

        # 垂直Transformer处理高频
        high_frequency = high_frequency.permute(0, 3, 2, 1).reshape(B * W, H, C)
        for layer in self.layers2:
            high_frequency = layer(high_frequency)
        high_frequency = high_frequency.reshape(B, W, H, C).permute(0, 3, 2, 1)

        # 融合原始特征+高低频特征
        fused_features = torch.cat([x, low_frequency, high_frequency], dim=1)
        x = self.fusion_conv(fused_features)
        return x


# 轴向上下文模块（最终整合）
class AxialContext_fft(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AxialContext_fft, self).__init__()
        self.HTransformer = HorizontalTransformerfft(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=embed_dim * 2,
            num_layers=1
        )
        self.VTransformer = VerticalTransformerfft(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=embed_dim,
            num_layers=1
        )
        self.fusion_conv = nn.Conv2d(3 * embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        Hcontext = self.HTransformer(x)
        Vcontext = self.VTransformer(x)
        fused_out = torch.cat([Hcontext, Vcontext, x], dim=1)
        fused_out = self.fusion_conv(fused_out)
        return fused_out

class AttentionModule(nn.Module):
    def __init__(self, visual_channels, linguistic_channels):
        super(AttentionModule, self).__init__()
        self.Wvi = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.Wvq = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.Wli = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wliv = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wo = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Vi, L):
        # 计算 Vim 和 Viq
        Vim = self.Wvi(Vi)
        Viq = self.Wvq(Vi)

        # 计算 Lik
        Lik = self.Wli(L)


        # 矩阵乘法，计算 Gi
        Viq_flat = Viq.view(Viq.size(0), Viq.size(1), -1)  # 展平
        Gi = torch.matmul(Viq_flat.permute(0, 2, 1), Lik)  # 计算点积
        Gi = self.softmax(Gi)  # 应用softmax

        # 计算 Liv
        Liv = self.Wliv(L)

        # 矩阵乘法，计算 Si
        Si = torch.matmul(Gi, Liv.permute(0, 2, 1))  # 矩阵乘法
        Si = Si.view(Vi.size())  # 恢复原始形状

        # 计算 Fi
        Fi = Vim * Si
        Fi = self.Wo(Fi)

        return Fi
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.embedding_1D = nn.Embedding(16, int(d_model))

    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        # learnable
        x = x + self.embedding_1D(torch.arange(16, device=device).to(device)).unsqueeze(1).repeat(1,x.size(1),  1)
        return self.dropout(x)

class Instruct_FHNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Instruct_FHNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = AxialContext(embed_dim=512, num_heads=8)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3]//2, filters[2]//2)
        self.gaussian_upsample4 = GaussianUpsample(filters[3]//2, filters[2]//2, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2]//2, filters[1]//2)
        self.gaussian_upsample3 = GaussianUpsample(filters[2]//2, filters[1]//2, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1]//2, filters[0]//2)
        self.gaussian_upsample2 = GaussianUpsample(filters[1]//2, filters[0]//2, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0]//2, filters[0]//2)
        self.gaussian_upsample1 = GaussianUpsample(filters[0]//2, filters[0]//2, kernel_size=3, scale_factor=2)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)
        x_gaussian = x[:, :C // 2, :, :]  # 高斯核部分
        x_decoder = x[:, C // 2:, :, :]  # 普通解码器部分

        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return out + skip_connection  # 加上跳跃连接



class AxialContextOnly(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=2048, num_layers=1):
        super(AxialContextOnly, self).__init__()
        # Horizontal Transformer (modeling along the width dimension)
        self.horizontal_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        # Vertical Transformer (modeling along the height dimension)
        self.vertical_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        # Fuse original, horizontal, and vertical features
        self.fusion_conv = nn.Conv2d(3 * embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # ------ Horizontal modeling (along W) ------
        x_h = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # [B*H, W, C]
        for layer in self.horizontal_layers:
            x_h = layer(x_h)
        x_h = x_h.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # ------ Vertical modeling (along H) ------
        x_v = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # [B*W, H, C]
        for layer in self.vertical_layers:
            x_v = layer(x_v)
        x_v = x_v.reshape(B, W, H, C).permute(0, 3, 2, 1)  # [B, C, H, W]

        # ------ Feature fusion ------
        fused = torch.cat([x, x_h, x_v], dim=1)  # [B, 3C, H, W]
        out = self.fusion_conv(fused)            # [B, C, H, W]
        return out

class LAVT_FHNet_AxialContext(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_FHNet_AxialContext, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = AxialContextOnly(embed_dim=512, num_heads=8)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3]//2, filters[2]//2)
        self.gaussian_upsample4 = GaussianUpsample(filters[3]//2, filters[2]//2, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2]//2, filters[1]//2)
        self.gaussian_upsample3 = GaussianUpsample(filters[2]//2, filters[1]//2, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1]//2, filters[0]//2)
        self.gaussian_upsample2 = GaussianUpsample(filters[1]//2, filters[0]//2, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0]//2, filters[0]//2)
        self.gaussian_upsample1 = GaussianUpsample(filters[0]//2, filters[0]//2, kernel_size=3, scale_factor=2)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)
        x_gaussian = x[:, :C // 2, :, :]  # 高斯核部分
        x_decoder = x[:, C // 2:, :, :]  # 普通解码器部分

        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return out + skip_connection  # 加上跳跃连接



class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

class LAVT_FHNet_CA_Block(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_FHNet_CA_Block, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = CA_Block(512,32,32)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3]//2, filters[2]//2)
        self.gaussian_upsample4 = GaussianUpsample(filters[3]//2, filters[2]//2, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2]//2, filters[1]//2)
        self.gaussian_upsample3 = GaussianUpsample(filters[2]//2, filters[1]//2, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1]//2, filters[0]//2)
        self.gaussian_upsample2 = GaussianUpsample(filters[1]//2, filters[0]//2, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0]//2, filters[0]//2)
        self.gaussian_upsample1 = GaussianUpsample(filters[0]//2, filters[0]//2, kernel_size=3, scale_factor=2)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)
        x_gaussian = x[:, :C // 2, :, :]  # 高斯核部分
        x_decoder = x[:, C // 2:, :, :]  # 普通解码器部分

        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return out + skip_connection  # 加上跳跃连接
class LAVT_FHNet_Dil(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_FHNet_Dil, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = Dblock(512)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3]//2, filters[2]//2)
        self.gaussian_upsample4 = GaussianUpsample(filters[3]//2, filters[2]//2, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2]//2, filters[1]//2)
        self.gaussian_upsample3 = GaussianUpsample(filters[2]//2, filters[1]//2, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1]//2, filters[0]//2)
        self.gaussian_upsample2 = GaussianUpsample(filters[1]//2, filters[0]//2, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0]//2, filters[0]//2)
        self.gaussian_upsample1 = GaussianUpsample(filters[0]//2, filters[0]//2, kernel_size=3, scale_factor=2)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)
        x_gaussian = x[:, :C // 2, :, :]  # 高斯核部分
        x_decoder = x[:, C // 2:, :, :]  # 普通解码器部分

        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return out + skip_connection  # 加上跳跃连接

class LAVT_FHNet_3_1(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_FHNet_3_1, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = AxialContext(embed_dim=512, num_heads=8)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3]//4*3, filters[2]//4*3)
        self.gaussian_upsample4 = GaussianUpsample(filters[3]//4, filters[2]//4, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2]//4*3, filters[1]//4*3)
        self.gaussian_upsample3 = GaussianUpsample(filters[2]//4, filters[1]//4, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1]//4*3, filters[0]//4*3)
        self.gaussian_upsample2 = GaussianUpsample(filters[1]//4, filters[0]//4, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0]//4*3, filters[0]//4*3)
        self.gaussian_upsample1 = GaussianUpsample(filters[0]//4, filters[0]//4, kernel_size=3, scale_factor=2)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)
        x_gaussian = x[:, :C // 4, :, :]  #
        x_decoder = x[:, C // 4:, :, :]  #
        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return out + skip_connection  # 加上跳跃连接



class LAVT_FHNet_1_3(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_FHNet_1_3, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = AxialContext(embed_dim=512, num_heads=8)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3]//4, filters[2]//4)
        self.gaussian_upsample4 = GaussianUpsample(filters[3]//4*3, filters[2]//4*3, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2]//4, filters[1]//4)
        self.gaussian_upsample3 = GaussianUpsample(filters[2]//4*3, filters[1]//4*3, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1]//4, filters[0]//4)
        self.gaussian_upsample2 = GaussianUpsample(filters[1]//4*3, filters[0]//4*3, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0]//4, filters[0]//4)
        self.gaussian_upsample1 = GaussianUpsample(filters[0]//4*3, filters[0]//4*3, kernel_size=3, scale_factor=2)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)

        x_decoder = x[:, :C // 4, :, :]  #
        x_gaussian = x[:, C // 4:, :, :]  #
        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return out + skip_connection  # 加上跳跃连接

class VisionAttention(nn.Module):
    def __init__(self, visual_channels, linguistic_channels):
        super(VisionAttention, self).__init__()

        self.Wvq = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.Wli = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wliv = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wo = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Vi, L):
        # 计算 Vim 和 Viq

        Viq = self.Wvq(Vi)

        # 计算 Lik
        Lik = self.Wli(L)


        # 矩阵乘法，计算 Gi
        Viq_flat = Viq.view(Viq.size(0), Viq.size(1), -1)  # 展平
        Gi = torch.matmul(Viq_flat.permute(0, 2, 1), Lik)  # 计算点积
        Gi = self.softmax(Gi)  # 应用softmax

        # 计算 Liv
        Liv = self.Wliv(L)

        # 矩阵乘法，计算 Si
        Si = torch.matmul(Gi, Liv.permute(0, 2, 1))  # 矩阵乘法
        Si = Si.view(Vi.size())  # 恢复原始形状

        # 计算 Fi
        Fi = Vi + Si
        Fi = self.Wo(Fi)

        return Fi

class Instruct_DLinkNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Instruct_DLinkNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


        self.Instruct1 = VisionAttention(64, 512)
        self.Instruct2 = VisionAttention(128, 512)
        self.Instruct3 = VisionAttention(256, 512)
        self.Instruct4 = VisionAttention(512, 512)

        self.context =  Dblock(512)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        #self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        #self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)

        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.context(e4)
        e44 = self.Instruct4(e4,instruct_feature)+e4


        # Decoder
        d4 = self.decoder4(e44) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


