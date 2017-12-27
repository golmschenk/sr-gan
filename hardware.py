"""
Code to simplify moving from CPU to GPU and vice versa.
"""

import torch


def gpu(element):
    """
    Moves the element to the GPU if available.

    :param element: The element to move to the GPU.
    :type element: torch.Tensor | torch.nn.Module
    :return: The element moved to the GPU.
    :rtype: torch.Tensor | torch.nn.Module
    """
    if torch.cuda.is_available():
        return element.cuda()
    else:
        return element


def cpu(element):
    """
    Moves the element to the CPU if GPU is available.

    :param element: The element to move to the CPU.
    :type element: torch.Tensor | torch.nn.Module
    :return: The element moved to the CPU.
    :rtype: torch.Tensor | torch.nn.Module
    """
    if torch.cuda.is_available():
        return element.cpu()
    else:
        return element


def load(model_path):
    """
    Loads a model, and if GPU is not available, insures that the model only loads onto CPU.

    :param model_path: The path to the model to be loaded.
    :type model_path: str
    :return: The loaded model.
    :rtype: dict[T]
    """
    if torch.cuda.is_available():
        return torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        return torch.load(model_path)
