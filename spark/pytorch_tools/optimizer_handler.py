import torch

__all__ = ["create_optimizer"]

def create_optimizer(model, learning_rate, optimizer_name, **kwargs):
    if optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == "Rprop":
        optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate, **kwargs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    return optimizer