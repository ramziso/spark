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
    elif scheduler == None:
        # Add StepLR that DO NOT change learning weight for
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma = 1.0)
    else:
        raise ValueError ("There is no learning scheduler like {}. Perhaps typing mistake?".format(scheduler))
    return lr_scheduler