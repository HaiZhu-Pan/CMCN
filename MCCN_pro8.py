# -*- coding: utf-8 -*-
# multiscale cross-attention convolutional network
# 日期：2023.10.13
# 使用金字塔多尺度卷积和交叉注意力机制，实现HSI和Lidar图像的特征融合
# 2023.10.16
# 改进方向：1.测试不同的激活函数，如Swish、ReLU等；2.测试不同的注意力机制，如Self-Attention、Cross-Attention等
# 3.cross-attention中的多头数和中间维度可以设置不同的值 4.可以设置可学习参数，alf*HSI_feature+beta*Lidar_feature
# 5.可以减少卷积层数，减少参数数量，提高计算效率
# 2023.10.20
# 总结一下这个网络：HSI用双分支多尺度卷积处理，生成光谱和空间特征
# Lidar用多尺度空间卷积处理，生成空间特征
# 用2个互注意力去处理生成的三个特征：HIS空间+HSI光谱，Lidar空间+HSI光谱
# 然后用1个互注意力处理这2个特征
# 然后cat这HIS空间+HSI光谱和刚才生成的那个特征
# 最后用一个分类模块去生成分类结果

# 2023年11月25,这是在论文中最终使用的版本
# 根据这个版本，后面写几个消融实验
# 2024.0728
# 在原有版本pro4的基础上，添加了参数共享

import torch
import torch.nn as nn
import torch.nn.functional as F
import attentions.crossview_attention as CrossAT

########################################################################################################################
# 定义Swish激活函数
class Swish(nn.Module):
    # def __init(self, inplace=True):
    def __init(self):
        super(Swish, self).__init__()
        # self.inplace = inplace

    def forward(self, x):
        # if self.inplace:
        #     x.mul_(torch.sigmoid(x))
        #     return x
        # else:
        return x * torch.sigmoid(x)


########################################################################################################################

class MCCN(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, midbands=24,multihead=1,reduction_ratio=2):
        super(MCCN, self).__init__()
        self.name = 'MCCN_pro8'
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.out_channels = out_channels
        ####################设置激活函数#######################################
        # self.activate = nn.PReLU()
        self.activate = nn.Mish()
        # self.activate = nn.GELU()
        # self.activate = Swish()

        #################### 处理HSI数据 #######################################
        ###### channel需要的layer
        kch = 1  # 进入网络的第一个channel核的大小
        kst = 1  # 进入网络的第一个channel的stride的大小
        inter_bands = ((self.in_channels1 - kch) // kst) + 1
        # midbands = 24

        # 降维
        self.conv01 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=midbands, kernel_size=(1, 1, kch), stride=(1, 1, kst)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 下面是三个不同的channel核，按照金字塔结构，分别对数据进行卷积
        self.conv101 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(1, 1, 7),
                                 padding=(0, 0, 3),
                                 stride=(1, 1, 1))
        self.conv102 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(1, 1, 5),
                                 padding=(0, 0, 2),
                                 stride=(1, 1, 1))
        self.conv103 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(1, 1, 3),
                                 padding=(0, 0, 1),
                                 stride=(1, 1, 1))
        # 这是卷积之后，对合并的数据进行BN和激活
        self.bn11 = nn.Sequential(
            nn.BatchNorm3d(midbands * 3 // 2, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 这是对合并数据进行降维，将到与输入数据一样的大小
        self.conv104 = nn.Sequential(
            nn.Conv3d(in_channels=midbands * 3 // 2, out_channels=midbands, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 这里是对one-shot合并的数据降维，降维到原来的数据大小
        self.conv105 = nn.Sequential(
            nn.Conv3d(in_channels=midbands * 3, out_channels=midbands, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 这里对Res数据处理成后面的Attention能处理的数据，去掉一个维度，这里是否要加BN？
        self.conv106 = nn.Sequential(
            nn.Conv3d(in_channels=midbands, out_channels=midbands, kernel_size=(1, 1, inter_bands), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )

        ##spatial需要的layer################
        # 数据进入spatial段的第一个卷积，目的是将样本的channel维度去掉，变为1
        self.conv21 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=midbands, kernel_size=(1, 1, in_channels1), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )

        # 下面是三个不同的kernel，按照金字塔结构，分别对空间维度进行卷积
        # 这里考虑7,5,3三种空间核，但是数据是7*7大小的，所以我们的Patch最好能大一点
        self.conv201 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(7, 7, 1),
                                 padding=(3, 3, 0),
                                 stride=(1, 1, 1))
        self.conv202 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(5, 5, 1),
                                 padding=(2, 2, 0),
                                 stride=(1, 1, 1))
        self.conv203 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(3, 3, 1),
                                 padding=(1, 1, 0),
                                 stride=(1, 1, 1))
        # 这是卷积之后，对合并数据进行BN和激活
        self.bn21 = nn.Sequential(
            nn.BatchNorm3d(midbands * 3 // 2, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 降维到数据一样的大小
        self.conv204 = nn.Sequential(
            nn.Conv3d(in_channels=midbands * 3 // 2, out_channels=midbands, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 这里是对one-shot合并的数据降维，降维到原来的数据大小
        self.conv205 = nn.Sequential(
            nn.Conv3d(in_channels=midbands * 3, out_channels=midbands, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 这里对Res数据处理成后面的Attention能处理的数据，去掉一个维度，这里是否要加BN？
        self.conv206 = nn.Sequential(
            nn.Conv3d(in_channels=midbands, out_channels=midbands, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )

        ## Lidar需要处理的layer
        self.conv31 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=midbands, kernel_size=(1, 1, in_channels2), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )

        # 下面是三个不同的kernel，按照金字塔结构，分别对空间维度进行卷积
        # 这里考虑7,5,3三种空间核，但是数据是7*7大小的，所以我们的Patch最好能大一点
        self.conv301 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(7, 7, 1),
                                 padding=(3, 3, 0),
                                 stride=(1, 1, 1))
        self.conv302 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(5, 5, 1),
                                 padding=(2, 2, 0),
                                 stride=(1, 1, 1))
        self.conv303 = nn.Conv3d(in_channels=midbands, out_channels=midbands // 2, kernel_size=(3, 3, 1),
                                 padding=(1, 1, 0),
                                 stride=(1, 1, 1))
        # 这是卷积之后，对合并数据进行BN和激活
        self.bn31 = nn.Sequential(
            nn.BatchNorm3d(midbands * 3 // 2, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 降维到数据一样的大小
        self.conv304 = nn.Sequential(
            nn.Conv3d(in_channels=midbands * 3 // 2, out_channels=midbands, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        # 这里是对one-shot合并的数据降维，降维到原来的数据大小
        # self.conv305 = nn.Sequential( #
        #     nn.Conv3d(in_channels=midbands * 3, out_channels=midbands, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        #     nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
        #     self.activate
        # )
        # # 这里对Res数据处理成后面的Attention能处理的数据，去掉一个维度，这里是否要加BN？
        # self.conv306 = nn.Sequential(
        #     nn.Conv3d(in_channels=midbands, out_channels=midbands, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        #     nn.BatchNorm3d(midbands, eps=0.001, momentum=0.1, affine=True),
        #     self.activate
        # )
        ## 注意力机制 #########################################################
        # 这里初始化的时候可以设置多头数和中间维度
        self.CrossAT1 = CrossAT.Crossview_attention(midbands,multihead,reduction_ratio)
        self.CrossAT2 = CrossAT.Crossview_attention(midbands,multihead,reduction_ratio)
        self.CrossAT3 = CrossAT.Crossview_attention(midbands, multihead,reduction_ratio)

        ###########classification################################################

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(2*midbands, eps=0.001, momentum=0.1, affine=True),
            self.activate
        )
        self.line1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2*midbands, out_channels)
        )
        # 定义一个dropout层
        # self.dropout = nn.Dropout(p=0.9)

    def HSI_spectral(self, x):
        # channel维处理 ###################
        x1 = self.conv01(x)
        # 一组金字塔卷积
        x101 = self.conv101(x1)
        x102 = self.conv102(x1)
        x103 = self.conv103(x1)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn11(xcat1)
        xp1 = self.conv104(xcat1)  # p表示金字塔

        x101 = self.conv101(xp1)
        x102 = self.conv102(xp1)
        x103 = self.conv103(xp1)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn11(xcat1)
        xp2 = self.conv104(xcat1)

        x101 = self.conv101(xp2)
        x102 = self.conv102(xp2)
        x103 = self.conv103(xp2)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn11(xcat1)
        xp3 = self.conv104(xcat1)

        # one-shot链接
        xshot1 = torch.cat((xp1, xp2, xp3), dim=1)
        xp4 = self.conv105(xshot1)
        # res
        xp5 = x1 + xp4
        # 降维
        xp6 = self.conv106(xp5)

        return xp6

    def HSI_spatial(self, x):
        # spatial维处理 ###################
        x1 = self.conv21(x)

        x101 = self.conv201(x1)
        x102 = self.conv202(x1)
        x103 = self.conv203(x1)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn21(xcat1)
        xp1 = self.conv204(xcat1)

        x101 = self.conv201(xp1)
        x102 = self.conv202(xp1)
        x103 = self.conv203(xp1)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn21(xcat1)
        xp2 = self.conv204(xcat1)

        x101 = self.conv201(xp2)
        x102 = self.conv202(xp2)
        x103 = self.conv203(xp2)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn21(xcat1)
        xp3 = self.conv204(xcat1)

        # one-shot
        xshot1 = torch.cat((xp1, xp2, xp3), dim=1)
        xp4 = self.conv205(xshot1)  # 这里从72降维到24
        # res
        xp5 = x1 + xp4
        # 降维
        xp6 = self.conv206(xp5)
        return xp6

    def Lidar_spatial(self, x):
        # spatial维处理 ###################
        x1 = self.conv31(x)

        x101 = self.conv301(x1)
        x102 = self.conv302(x1)
        x103 = self.conv303(x1)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn31(xcat1)
        xp1 = self.conv304(xcat1)

        x101 = self.conv301(xp1)
        x102 = self.conv302(xp1)
        x103 = self.conv303(xp1)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn31(xcat1)
        xp2 = self.conv304(xcat1)

        x101 = self.conv301(xp2)
        x102 = self.conv302(xp2)
        x103 = self.conv303(xp2)
        xcat1 = torch.cat((x101, x102, x103), dim=1)
        xcat1 = self.bn31(xcat1)
        xp3 = self.conv304(xcat1)

        # one-shot
        xshot1 = torch.cat((xp1, xp2, xp3), dim=1)
        xp4 = self.conv205(xshot1)  # 这里从72降维到24
        # res
        xp5 = x1 + xp4
        # 降维
        xp6 = self.conv206(xp5)
        return xp6

    def forward(self, x1, x2):
        ## x1对应的HSI图像处理部分 ######################################
        x11 = self.HSI_spatial(x1)  # spatial wise
        x12 = self.HSI_spectral(x1)  # spectral wise

        # 消除第四维度
        x11 = torch.squeeze(x11, dim=4)
        x12 = torch.squeeze(x12, dim=4)

        ## x2对应Lidar图像处理部分
        x21 = self.Lidar_spatial(x2)
        # 消除第四维度
        x21 = torch.squeeze(x21, dim=4)

        ## cross attention
        xfuse1 = self.CrossAT1(x12, x11, x11)[0]
        xfuse2 = self.CrossAT2(x12, x21, x21)[0]
        xfuse3 = self.CrossAT3(xfuse2, xfuse1, xfuse1)[0]

        # 合并融合特征
        x31 = torch.cat((xfuse1,xfuse3), dim=1)  # 32*48*7*7

        # 分类结果
        x31 = self.avgpool(x31)
        x31 = x31.squeeze(-1).squeeze(-1)
        # 这里放一个dropout
        # x31 = self.dropout(x31)

        out = self.line1(x31)

        return out


# 测试网络

from thop import profile
from thop import clever_format
if __name__ == '__main__':
    model = MCCN(63, 1, 6, 24,2,2)  # tre: 63,6   MUU: 64,11    Hou: 144,15
    model.eval()
    print(model)
    input1 = torch.randn(32, 1, 7, 7, 63)
    input2 = torch.randn(32, 1, 7, 7, 1)
    x = model(input1, input2)
    print(x.size())
    # 输出flops和params
    flops,params=profile(model, inputs=(input1, input2))
    macs, params = clever_format([flops, params], "%.2f")
    print( params,macs)

    print("get the result!")
