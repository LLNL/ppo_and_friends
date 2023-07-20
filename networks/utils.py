"""
    A home for network utilities.
"""
import torch.nn as nn
import numpy as np
from functools import reduce

from mpi4py import MPI
comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_procs = comm.Get_size()

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


def create_sequential_network(
    in_size,
    out_size,
    hidden_size,
    hidden_depth,
    activation,
    out_init = None):
    """
        Create a Sequential torch network.

        Arguments:
            in_size         The size of the input field (int).
            out_size        The size of the output field (int).
            activation      The activation function to use on the output
                            of hidden layers.
            hidden_size     Can either be an int or list of ints. If an int,
                            all layers will be this size. Otherwise, a list
                            designates the size for each layer. Note that
                            the hidden_depth argument is ignored if this
                            argument is a list and the depth is instead
                            taken from the length of the list. Note that
                            this argument can be set to 0 or an empty list,
                            resulting in only an input and output layer.
            hidden_depth    The number of hidden layers. Note that this is
                            ignored if hidden_size is a list.
            out_init        A std weight to apply to the output layer.
    """
    if type(hidden_size) != list:

        if ((hidden_size == 0 and hidden_depth != 0) or
            (hidden_size != 0 and hidden_depth == 0)):

            msg  = "ERROR: if either hidden_size or hidden_depth "
            msg += "is 0, both must be 0,"
            msg += "but received "
            msg += "hidden_size of {} ".format(hidden_size)
            msg += "and hidden_depth of {}.".format(hidden_depth)
            rank_print(msg)
            comm.Abort()

        hidden_size = [hidden_size] * hidden_depth
    else:
        hidden_depth = len(hidden_size)

    layers = []

    if len(hidden_size) != 0:

        layers.append(init_layer(nn.Linear(in_size, hidden_size[0])))
        layers.append(activation)

        inner_layer_list = []

        for i in range(hidden_depth - 1):
            inner_layer_list.append(init_layer(
                nn.Linear(
                    hidden_size[i],
                    hidden_size[i + 1])))

            inner_layer_list.append(activation)

        layers.append(nn.Sequential(*inner_layer_list))

        if out_init != None:
            layers.append(init_layer(
                nn.Linear(hidden_size[-1], out_size),
                    weight_std=out_init))
        else:
            layers.append(init_layer(
                nn.Linear(hidden_size[-1], out_size)))
    else:
        if out_init != None:
            layers.append(init_layer(nn.Linear(in_size, out_size),
                weight_std=out_init))
        else:
            layers.append(init_layer(nn.Linear(in_size, out_size)))

    return nn.Sequential(*layers)


def get_size_and_shape(descriptor):
    """
        Given a shape/size descriptor as either a tuple or int,
        return the associated shape and size.

        Arguments:
            descriptor    An int or tuple representing the size/shape.

        Returns:
            The size and shape as (int, tuple).
    """
    assert type(descriptor) == tuple or type(descriptor) == int

    if type(descriptor) == tuple:
        size  = reduce(lambda a, b: a*b, descriptor)
        shape = descriptor
    else:
        size  = descriptor
        shape = (descriptor,)

    return size, shape
