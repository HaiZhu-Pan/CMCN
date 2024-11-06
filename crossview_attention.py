# Copyright (c) Gorilla-Lab. All rights reserved.
# 改写自：VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention
# 去掉一个形状注意力

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class Crossview_attention(nn.Module):
    def __init__(self, input_channels: int, numhead: int = 1, reduction_ratio=2):
        r"""
        Args:
            input_channels (int): input channel of conv attention
            numhead (int, optional): the number of attention heads. Defaults to 1.
        """
        super().__init__()
        # self.q_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, 3, 1, 1)
        self.q_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)

        # self.k_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, 3, 1, 1)
        #
        self.k_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)

        # self.v_conv = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
        self.v_conv = nn.Conv2d(input_channels, input_channels, 1, 1, 0)
        self.out_sem_conv = nn.Conv2d(input_channels, input_channels, 1, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.channels = input_channels // reduction_ratio
        self.numhead = numhead
        self.head_dim = self.channels // numhead
        self.sem_norm = nn.LayerNorm(input_channels)


    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                ):
        r"""
        Args:
            query (torch.Tensor, [B, C, H_qk, W_qk]): feature of query
            key (torch.Tensor, [B, C, H_qk, W_qk]): feature of key
            value (torch.Tensor, [B, C, H_v, W_v]): feature of value
            q_pos_emb (Optional[torch.Tensor], optional, [[B, C, H_q, W_q]]):
                positional encoding. Defaults to None.
            k_pos_emb (Optional[torch.Tensor], optional, [[B, C, H_kv, W_kv]]):
                positional encoding. Defaults to None.
        """

        view = query + 0  # NOTE: a funny method to deepcopy
        input_channel = view.shape[1]

        # to qkv forward
        # q = self.q_conv(query)
        q = self.q_sem_conv(query)

        # k = self.k_conv(key)
        k = self.k_sem_conv(key)

        v = self.v_conv(value)

        # read shape of qkv
        bs = q.shape[0]
        qk_channel = q.shape[1]  # equal to the channel of `k`
        v_channel = v.shape[1]  # channel of `v`
        h_q, w_q = q.shape[2:]  # height and weight of query map
        h_kv, w_kv = k.shape[2:]  # height and weight of key and value map
        numhead = self.numhead
        qk_head_dim = qk_channel // numhead
        v_head_dim = v_channel // numhead

        # scale query
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling

        # reshape(sequentialize) qkv
        q = rearrange(q, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_q, w=w_q)
        q = rearrange(q, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_q, w=w_q, d=qk_head_dim)
        # .contiguous()方法的作用是返回一个连续的张量，即重新分配内存空间，并将张量的数据按照连续的方式进行存储。
        # 这样可以确保张量在内存中的布局是连续的，从而提高了访问和计算的效率
        q = q.contiguous()
        k = rearrange(k, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_kv, w=w_kv)
        k = rearrange(k, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_kv, w=w_kv, d=qk_head_dim)
        k = k.contiguous()
        v = rearrange(v, "b c h w -> b c (h w)", b=bs, c=v_channel, h=h_kv, w=w_kv)
        v = rearrange(v, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                      n=numhead, h=h_kv, w=w_kv, d=v_head_dim)
        v = v.contiguous()

        # get the attention map
        # 这是矩阵乘法的结果，是注意力的值，但是有负数，在下面用softmax把它变成概率
        energy = torch.bmm(q, k.transpose(1, 2))  # [h_q*w_q, h_kv*w_kv]，矩阵乘法，（32,49,51）bmm（32,51,49）=（32,49,49）
        attention = F.softmax(energy, dim=-1)  # [h_q*w_q, h_kv*w_kv]

        # get the attention output
        r = torch.bmm(attention, v)  # [bs * nhead, h_q*w_q, C']
        r = rearrange(r, "(b n) (h w) d -> b (n d) h w", b=bs,
                      n=numhead, h=h_q, w=w_q, d=v_head_dim)
        r = r.contiguous()
        r = self.out_sem_conv(r)

        # residual
        temp_view = view + r
        temp_view = temp_view.view(bs, input_channel, -1).permute(2, 0, 1).contiguous()
        temp_view = self.sem_norm(temp_view)

        #回复原来的形状
        temp_view=temp_view.permute(1,2,0)
        output=temp_view.view(bs,input_channel,h_q, w_q)

        return output, attention

def main():
    # 第一个参数：输入数据的维度，是序号：1 位置的数值，例如输入为(8,80, 7, 7)，这里就是80,三个参数的维度要求相同
    # 第二个参数：多头注意力的头的数量：默认是1，可以是大于一的数，要注意，能被第一个参数或根据第三个参数调整之后的维度整除才可以
    # 第三个参数：将输入数据的维度降维，这里应该是要简化计算才使用的，默认是2，也可以用大于2的值，注意要能整除
    net = Crossview_attention(80,4,1)
    x = torch.randn(8,80, 7, 7)

    outputs, attentions = net(x,x,x)  #这里三个参数分别对应q,k,v,且k=v
    # out = net(tmp)
    print(outputs.shape)

    # net = Dilated_conv_block(8)
    # tmp = torch.randn(32, 32, 1)
    # out = net(tmp)
    # print(out.shape)


if __name__ == '__main__':
    main()