import torch

__all__ = ["create_loss_func"]

def create_loss_func(loss_func_name = "CrossEntropyLoss", **kwargs):
    if loss_func_name == "NLLLoss":
        loss_func = torch.nn.NLLLoss(**kwargs)
    elif loss_func_name == "CrossEntropyLoss":
        loss_func = torch.nn.CrossEntropyLoss(**kwargs)
    return loss_func