# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Tuple, Union

import torch
import torch.nn as nn


class SymQuantizer(torch.autograd.Function):
    """Symmetric linear quantisation"""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, clip_val: torch.Tensor, num_bits: int, layerwise: bool) -> torch.Tensor:
        ctx.save_for_backward(input, clip_val)
        input = torch.clamp(input, clip_val[0], clip_val[1])

        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError(
                    f"Unsupported tensor size for quantization. Expected <4 dimensions, got {input.ndimension()}."
                )

        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx: saved non-clipped full-precision tensor and clip_val
            grad_output: gradient wrt the quantized tensor
        Returns:
            grad_input: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # un-clipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN). Ref: https://arxiv.org/abs/1605.04711"""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, clip_val: torch.Tensor, num_bits: int, layerwise: bool):
        ctx.save_for_backward(input, clip_val)

        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])

        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:  # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx: saved non-clipped full-precision tensor and clip_val
            grad_output: gradient wrt the quantized tensor
        Returns:
            grad_input: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # un-clipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_act: bool = True,
        input_bits: int = 8,
        weight_bits: int = 2,
        clip_val: float = 2.5,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits
        self.quantize_act = quantize_act
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.register_buffer("weight_clip_val", torch.tensor([-clip_val, clip_val]))
        if self.quantize_act:
            self.input_bits = input_bits
            self.act_quantizer = SymQuantizer
            self.register_buffer("act_clip_val", torch.tensor([-clip_val, clip_val]))

    def forward(self, input):
        # quantize weight
        weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        # quantize input
        input = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        out = nn.functional.linear(input, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx=None, weight_bits: int = 2, clip_val: float = 2.5
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight_bits = weight_bits
        self.layerwise = False

        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        else:
            self.weight_quantizer = SymQuantizer

        self.register_buffer("weight_clip_val", torch.tensor([-clip_val, clip_val]))

    def forward(self, input):
        weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits,self.layerwise)
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out


class QuantizeConv(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[str, Union[int, Tuple[int]]] = 0,
        bias: bool = True,
        quantize_act: bool = True,
        input_bits: int = 8,
        weight_bits: int = 2,
        clip_val: float = 2.5,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits
        self.quantize_act = quantize_act
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.register_buffer("weight_clip_val", torch.tensor([-clip_val, clip_val]))
        if self.quantize_act:
            self.input_bits = input_bits
            self.act_quantizer = SymQuantizer
            self.register_buffer("act_clip_val", torch.tensor([-clip_val, clip_val]))

    def forward(self, input):
        # quantize weight
        weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        # quantize input
        input = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        out = nn.functional.conv1d(input, weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1).expand_as(out)

        return out
