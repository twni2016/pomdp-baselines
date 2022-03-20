"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from torchkit import pytorch_utils as ptu
from torchkit.core import PyTorchModule
from torchkit.modules import LayerNorm


class Mlp(PyTorchModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=ptu.identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along last dim
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


import numpy as np


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_shape,
        embed_size=100,
        depth=8,
        kernel_size=(2, 2),
        stride=1,
        from_flattened=False,
        normalize_pixel=False,
    ):
        super(ImageEncoder, self).__init__()
        self.shape = image_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth

        self.conv1 = nn.Conv2d(image_shape[0], depth, kernel_size, stride)
        self.conv2 = nn.Conv2d(depth, 2 * depth, kernel_size, stride)
        self.linear = nn.Linear(self.conv_out_size(), embed_size)
        self.activation = nn.ReLU()

        self.from_flattened = from_flattened
        self.normalize_pixel = normalize_pixel
        self.embed_size = embed_size

    def forward(self, image):
        # image of size [N, C, H, W]
        # return embedding of shape [N, embed_size]
        if self.from_flattened:
            batch_size = image.shape[:-1]
            img_shape = [np.prod(batch_size)] + list(self.shape)
            image = torch.reshape(image, img_shape)
        else:
            batch_size = [image.shape[0]]
        
        if self.normalize_pixel:
            image = image / 255.0

        embed = self.conv1(image)
        embed = self.activation(embed)
        embed = self.conv2(embed)
        embed = self.activation(embed)
        embed = torch.reshape(embed, list(batch_size) + [-1])
        embed = self.linear(embed)
        return embed

    def conv_out_size(self):
        h_w = self.shape[-2:]
        out_h_w = conv_output_shape(h_w, self.kernel_size, self.stride)
        out_h_w = conv_output_shape(out_h_w, self.kernel_size, self.stride)

        return out_h_w[0] * out_h_w[1] * self.depth * 2
