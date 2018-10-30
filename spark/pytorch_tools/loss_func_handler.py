import torch

__all__ = ["create_loss_func"]

def create_loss_func(loss_func_name = "CrossEntropyLoss", class_imbalance = None):
    if loss_func_name == "NLLLoss":
        loss_func = torch.nn.NLLLoss()
    elif loss_func_name == "CrossEntropyLoss":
        loss_func = torch.nn.CrossEntropyLoss()
    return loss_func