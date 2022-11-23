"""
    A home for network utilities.
"""
import torch.nn as nn
import numpy as np

def get_conv2d_out_size(in_size,
                        padding,
                        kernel_size,
                        stride):
    """
        Get the output size of a 2d convolutional layer.

        Arguments:
            in_size        The input size.
            padding        The padding used.
            kernel_size    The kernel size used.
            stride         The stride used.

        Returns:
            The expected output size of the convolution.
    """
    out_size = int(((in_size + 2.0 * padding - (kernel_size - 1) - 1)\
        / stride) + 1)
    return out_size


def get_maxpool2d_out_size(in_size,
                           padding,
                           kernel_size,
                           stride):
    """
        Get the output size of a 2d max pool layer.

        Arguments:
            in_size        The input size.
            padding        The padding used.
            kernel_size    The kernel size used.
            stride         The stride used.

        Returns:
            The expected output size of the max pool.
    """
    return get_conv2d_out_size(in_size, padding, kernel_size, stride)


def init_layer(layer,
               weight_std = np.sqrt(2),
               bias_const = 0.0):
    """
        Orthogonally initialize a neural network layer using an std weight and
        a bias constant.

        Arguments:
            layer         The network layer.
            weight_std    The std weight.
            bias_const    The bias constants.

        Returns:
            The initialized layer.
    """

    nn.init.orthogonal_(layer.weight, weight_std)
    nn.init.constant_(layer.bias, bias_const)

    return layer

def init_net_parameters(net,
                        weight_std = np.sqrt(2),
                        bias_const = 0.0):
    """
        Orthogonally initialize a neural network using an std weight and
        a bias constant.

        Arguments:
            net           The network.
            weight_std    The std weight.
            bias_const    The bias constants.

        Returns:
            The initialized network.
    """

    for name, param in net.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param, weight_std)
        elif 'bias' in name:
            nn.init.constant_(param, bias_const)

    return net
