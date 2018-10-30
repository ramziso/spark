import numpy as np

__all__ = ["trainable_parameters",
            "trainable_parameters_num",
            "total_parameters",
            "total_parameters_num",
            "WeightHistogram"]

def trainable_parameters(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))

def trainable_parameters_num(model):
    return sum([np.prod(p.size()) for p in trainable_parameters(model)])

def total_parameters(model):
    return list(model.parameters())

def total_parameters_num(model):
    return sum([np.prod(p.size()) for p in total_parameters(model)])

def WeightHistogram(model, target="trainable"):
    if target == "trainable":
        params = trainable_parameters(model)
    elif target == "all":
        params = total_parameters(model)

    layer_parameters = []
    for layer_parameter in params:
        layer_parameter = layer_parameter.reshape(1,-1).detach()[0].numpy()
        layer_parameters.append(layer_parameter)
    layer_parameters = np.concatenate(layer_parameters)

    return np.histogram(layer_parameters)

# Testing code
#from torchvision.models import resnet18
#model = resnet18(False)
#histogram = WeightHistogram(model)