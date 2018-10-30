import torch

__all__ = ["print_child_layers",
           "print_layer_parameters",
           "finetuning_all_layer",
           "train_last_layer"]

def print_child_layers(model):
    layer_counter = 0
    for layer in model.children():
        #print(" layer", layer_counter, "is -")
        print(layer)
        layer_counter += 1
    return layer_counter

def print_layer_parameters(model, layer_index):
    child_list = list(model.children())
    for parameters in child_list[layer_index].parameters():
        #print("{}th layers current parameter : \n",parameters)
        break

def finetuning_all_layer(model, target_layers = None):
    child_list = list(model.children())
    child_len = len(child_list)
    target_layers = target_layers
    if target_layers == None:
        target_layers = range(child_len)
    else:
        target_layers = range(target_layers[0], target_layers[1])

    child_counter = 0
    for child in model.children():
        if child_counter in target_layers:
            for param in child.parameters():
                param.requires_grad = False
        elif child_counter not in target_layers:
            for param in child.parameters():
                param.requires_grad = True

        child_counter += 1
    return model

def train_last_layer(model, target_layers = None):
    child_counter = 0
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
        child_counter += 1
    return model

# after you froze some layers in the network, you must reset the optimizer like next:
#optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)