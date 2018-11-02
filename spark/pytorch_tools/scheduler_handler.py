import torch.optim as optim

__all__ = ["create_train_scheduler"]

def __find_torch_object(scheduler):
    if isinstance(scheduler, []):
        pass

def create_train_scheduler(optimizer, scheduler, **kwargs):
    if scheduler == "LambdaLR":
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, **kwargs)
    elif scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnelingLR(optimizer, **kwargs)
    elif scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    return lr_scheduler