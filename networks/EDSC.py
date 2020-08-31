import torch
import torchvision
import cv2
import os
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torchvision.models as models
import math
import sys
import torch.nn.functional as F
from networks import sepconv



class Efficient_HetConv2d(nn.Module):
    def __init__(self, in_feats, out_feats, p=4, ks=3, pad=1):
        super(Efficient_HetConv2d, self).__init__()
        if in_feats % p != 0:
            raise ValueError('in_channels must be divisible by p')
        if out_feats % p != 0:
            raise ValueError('out_channels must be divisible by p')
        self.conv3x3 = nn.Conv2d(in_feats, out_feats, kernel_size=ks, padding=pad, groups=p)
        self.conv1x1_ = nn.Conv2d(in_feats, out_feats, kernel_size=1, groups=p)
        self.conv1x1 = nn.Conv2d(in_feats, out_feats, kernel_size=1)

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) - self.conv1x1_(x)


class Network(nn.Module):
    '''
    Network of EDSC
    '''
    def __init__(self, Generated_ks=5, Het_p=4, useBias=True, isMultiple=False):
        super(Network, self).__init__()
        self.het_p = Het_p
        self.generated_ks = Generated_ks
        self.useBias = useBias
        self.isMultiple = isMultiple
        if self.isMultiple:
            self.estimator_in = 65
        else:
            self.estimator_in = 64

        def Basic(intInput, intOutput, ks, pad):
            return torch.nn.Sequential(
                Efficient_HetConv2d(in_feats=intInput, out_feats=intOutput, p=self.het_p, ks=ks, pad=pad),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(in_feats=intOutput, out_feats=intOutput, p=self.het_p, ks=ks, pad=pad),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(in_feats=intOutput, out_feats=intOutput, p=self.het_p, ks=ks, pad=pad),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Efficient_HetConv2d(in_feats=intOutput, out_feats=intOutput, p=self.het_p, ks=3, pad=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def KernelNet():
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks, out_channels=self.generated_ks, kernel_size=3, stride=1, padding=1)
                )
        # end

        def Offsetnet():
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks ** 2, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1)
                )


        def Masknet():
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks ** 2, out_channels=self.generated_ks ** 2, kernel_size=3,
                             stride=1, padding=1),
                    torch.nn.Sigmoid()
                )
                

        def Biasnet():
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
                )
                

        self.moduleConv1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(in_feats=32, out_feats=32, p=self.het_p, ks=3, pad=1),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(in_feats=32, out_feats=32, p=self.het_p, ks=3, pad=1),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv2 = Basic(32, 64, 3, 1)
        self.moduleConv3 = Basic(64, 128, 3, 1)
        self.moduleConv4 = Basic(128, 256, 3, 1)
        self.moduleConv5 = Basic(256, 512, 3, 1)

        self.moduleDeconv5 = Basic(512, 512, 3, 1)
        self.moduleDeconv4 = Basic(512, 256, 3, 1)
        self.moduleDeconv3 = Basic(256, 128, 3, 1)
        self.moduleDeconv2 = Basic(128, 64, 3, 1)

        self.moduleUpsample5 = Upsample(512, 512)
        self.moduleUpsample4 = Upsample(256, 256)
        self.moduleUpsample3 = Upsample(128, 128)
        self.moduleUpsample2 = Upsample(64, 64)

        self.moduleVertical1 = KernelNet()
        self.moduleVertical2 = KernelNet()
        self.moduleHorizontal1 = KernelNet()
        self.moduleHorizontal2 = KernelNet()

        self.moduleOffset1x = Offsetnet()
        self.moduleOffset1y = Offsetnet()
        self.moduleOffset2x = Offsetnet()
        self.moduleOffset2y = Offsetnet()

        self.moduleMask1 = Masknet()
        self.moduleMask2 = Masknet()

        self.moduleBias = Biasnet()
        # self.moduleOffset1x.register_backward_hook(self._set_lr)

    # end
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, tensors):
        def AdaptiveDC(TensorInput, Offsetx, Offsety, Mask):
            '''
            Adaptive Deformable Convolution
            :param TensorInput: inputframe
            :param Offsetx: offset x
            :param Offsety: offset y
            :param Mask: mask for each pixel
            :return: bilnear interpolated point of a pixel, a ks times bitter image
            '''

            def _get_p_n(kernel_size, N, dtype):
                p_n_x, p_n_y = torch.meshgrid(
                    torch.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1),
                    torch.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1))

                p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
                # p_n = [-1, -1, -1,  0,  0,  0,  1,  1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1]
                p_n = p_n.contiguous().view(1, 2 * N, 1, 1).type(dtype)
                # changed into a 4 dimensional tensor, which is returned in this function

                return p_n

            def _get_p_0(stride, h, w, N, dtype):
                p_0_x, p_0_y = torch.meshgrid(
                    torch.arange(1, h * stride + 1, stride),
                    torch.arange(1, w * stride + 1, stride))

                p_0_x = torch.flatten(p_0_x).contiguous().view(1, 1, h, w).repeat(1, N, 1, 1)
                p_0_y = torch.flatten(p_0_y).contiguous().view(1, 1, h, w).repeat(1, N, 1, 1)
                #  shape = (Batch, N, h, w)
                p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

                return p_0  # shape = (Batch, 2*N, h, w)

            def _get_p(offset, dtype):
                N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

                # (1, 2N, 1, 1)
                p_n = _get_p_n(int(math.sqrt(N)), N, dtype)
                # (1, 2N, h, w)
                p_0 = _get_p_0(1, h, w, N, dtype)
                p = p_0 + p_n + offset
                return p

            def _get_x_q(x, q, N):
                b, h, w, _ = q.size()
                padded_w = x.size(3)
                c = x.size(1)
                # (b, c, h*w)
                x = x.contiguous().view(b, c, -1)

                # (b, h, w, N)
                index = q[..., :N] * padded_w + q[...,
                                                N:]  # offset_x*w + offset_y, due to the same operation of each channel, the shape of index has no channel dimension
                # (b, c, h*w*N)
                index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

                x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

                return x_offset

            def _reshape_x_offset(x_offset, ks):
                b, c, h, w, N = x_offset.size()
                x_offset = torch.cat(
                    [x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
                x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

                return x_offset

            N = Offsetx.size(1)
            ks = int(math.sqrt(N))
            p = _get_p(torch.cat([Offsetx, Offsety], 1), Offsetx.data.type())
            # (b, h, w, 2N)
            p = p.contiguous().permute(0, 2, 3, 1)

            q_lt = p.detach().floor()  # both of the x,y coordinate should be floored
            q_rb = q_lt + 1

            q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, TensorInput.size(2) - 1),
                              torch.clamp(q_lt[..., N:], 0, TensorInput.size(3) - 1)],
                             dim=-1).long()  # keep q in in the frame.
            q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, TensorInput.size(2) - 1),
                              torch.clamp(q_rb[..., N:], 0, TensorInput.size(3) - 1)], dim=-1).long()
            q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
            q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

            # clip p
            p = torch.cat([torch.clamp(p[..., :N], 0, TensorInput.size(2) - 1),
                           torch.clamp(p[..., N:], 0, TensorInput.size(3) - 1)], dim=-1)  # float

            # bilinear kernel (b, h, w, N)
            g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
            g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
            g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
            g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

            # (b, c, h, w, N)
            x_q_lt = _get_x_q(TensorInput, q_lt, N)
            x_q_rb = _get_x_q(TensorInput, q_rb, N)
            x_q_lb = _get_x_q(TensorInput, q_lb, N)
            x_q_rt = _get_x_q(TensorInput, q_rt, N)

            # (b, c, h, w, N)
            x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(
                dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt

            m = Mask.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
            x_offset = _reshape_x_offset(x_offset, ks)

            return x_offset


        tensorFirst = tensors[0]
        tensorSecond = tensors[1]
        if self.isMultiple:
            tensorTime = tensors[2]

        tensorConv1 = self.moduleConv1(torch.cat([tensorFirst, tensorSecond], 1))
        tensorConv2 = self.moduleConv2(torch.nn.functional.avg_pool2d(input=tensorConv1, kernel_size=2, stride=2))
        tensorConv3 = self.moduleConv3(torch.nn.functional.avg_pool2d(input=tensorConv2, kernel_size=2, stride=2))
        tensorConv4 = self.moduleConv4(torch.nn.functional.avg_pool2d(input=tensorConv3, kernel_size=2, stride=2))
        tensorConv5 = self.moduleConv5(torch.nn.functional.avg_pool2d(input=tensorConv4, kernel_size=2, stride=2))

        tensorDeconv5 = self.moduleUpsample5(
            self.moduleDeconv5(torch.nn.functional.avg_pool2d(input=tensorConv5, kernel_size=2, stride=2)))
        tensorDeconv4 = self.moduleUpsample4(self.moduleDeconv4(tensorDeconv5 + tensorConv5))
        tensorDeconv3 = self.moduleUpsample3(self.moduleDeconv3(tensorDeconv4 + tensorConv4))
        tensorDeconv2 = self.moduleUpsample2(self.moduleDeconv2(tensorDeconv3 + tensorConv3))

        tensorCombine = tensorDeconv2 + tensorConv2

        tensorFirst = torch.nn.functional.pad(input=tensorFirst,
                                              pad=[int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0)),
                                                   int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0))],
                                              mode='replicate')
        tensorSecond = torch.nn.functional.pad(input=tensorSecond,
                                               pad=[int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0)),
                                                    int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0))],
                                               mode='replicate')


        if self.isMultiple:
            v1 = self.moduleVertical1(torch.cat([tensorCombine, tensorTime], 1))
            v2 = self.moduleVertical2(torch.cat([tensorCombine, 1. - tensorTime], 1))
            h1 = self.moduleHorizontal1(torch.cat([tensorCombine, tensorTime], 1))
            h2 = self.moduleHorizontal2(torch.cat([tensorCombine, 1. - tensorTime], 1))
            # resampled frames guided by offsets and masks
            tensorSampled1 = AdaptiveDC(tensorFirst, self.moduleOffset1x(torch.cat([tensorCombine, tensorTime], 1)),
                                        self.moduleOffset1y(torch.cat([tensorCombine, tensorTime], 1)),
                                        self.moduleMask1(torch.cat([tensorCombine, tensorTime], 1)))
            tensorSampled2 = AdaptiveDC(tensorSecond,
                                        self.moduleOffset2x(torch.cat([tensorCombine, 1. - tensorTime], 1)),
                                        self.moduleOffset2y(torch.cat([tensorCombine, 1. - tensorTime], 1)),
                                        self.moduleMask2(torch.cat([tensorCombine, 1. - tensorTime], 1)))
        else:
            v1 = self.moduleVertical1(tensorCombine)
            v2 = self.moduleVertical2(tensorCombine)
            h1 = self.moduleHorizontal1(tensorCombine)
            h2 = self.moduleHorizontal2(tensorCombine)
            offset1x = self.moduleOffset1x(tensorCombine)
            offset1y = self.moduleOffset1y(tensorCombine)
            offset2x = self.moduleOffset2x(tensorCombine)
            offset2y = self.moduleOffset2y(tensorCombine)
            mask1 = self.moduleMask1(tensorCombine)
            mask2 = self.moduleMask2(tensorCombine)

            # resampled frames guided by offsets and masks
            tensorSampled1 = AdaptiveDC(tensorFirst, offset1x,
                                        offset1y,
                                        mask1)
            tensorSampled2 = AdaptiveDC(tensorSecond, offset2x,
                                        offset2y,
                                        mask2)

        tensorDot1 = sepconv.FunctionSepconv(tensorInput=tensorSampled1, tensorVertical=v1, tensorHorizontal=h1)
        tensorDot2 = sepconv.FunctionSepconv(tensorInput=tensorSampled2, tensorVertical=v2, tensorHorizontal=h2)

        if self.useBias:
                return tensorDot1 + tensorDot2 + self.moduleBias(tensorCombine)
        else:
                return tensorDot1 + tensorDot2
            

    # end forward
# end class
