# -*- coding: future_fstrings -*-
import torch
import numpy as np
import os
import torch.nn.functional as F


def identity(x):
    return x


def id_to_onehot(id, n_classes):
    """

    :param id: arr/tensor of size (n, 1)
    :param n_classes: int
    :return: one hot vector of size
    """
    one_hot = zeros((id.shape[0], n_classes))
    one_hot[torch.arange(one_hot.shape[0]), id[:, 0]] = 1
    return one_hot


def cross_entropy_one_hot(source, target, reduction="none"):
    _, labels = target.max(dim=-1)  # probabilities are on last dimension
    return F.cross_entropy(source, labels, reduction=reduction)


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(elem_or_tuple_to_variable(e) for e in elem_or_tuple)
    return from_numpy(elem_or_tuple)


def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: elem_or_tuple_to_variable(x)
        for k, x in filter_batch(np_batch)
        if x.dtype != np.dtype("O")  # ignore object (e.g. dictionaries)
    }


def list_from_numpy(li):
    "convert all elements in input list to torch"
    return [from_numpy(element) for element in li]


"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device(f"cuda:{gpu_id}" if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to("cpu").detach().numpy()


def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)


def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs).to(device)


def randn_like(*args, **kwargs):
    return torch.randn_like(*args, **kwargs).to(device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)


def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs).to(device)


def round_tensor(tensor, n_digits):
    return (tensor * 10 ** n_digits).round() / (10 ** n_digits)
