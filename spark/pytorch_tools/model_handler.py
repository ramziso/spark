import torch

def create_model_object(model_info):
    if model_info["pretrained"] == "imagenet":
        model = model_info["model_constructer"](input_size=model_info["input_size"],
                                                num_classes=model_info["num_classes"],
                                                pretrained=model_info["pretrained"])

    elif model_info["pretrained"] != None:
        model = model_info["model_constructer"](input_size=model_info["input_size"],
                                                num_classes=model_info["num_classes"],
                                                pretrained=None)
        checkpoint = torch.load(model_info["pretrained"])
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = model_info["model_constructer"](input_size=model_info["input_size"],
                                                num_classes=model_info["num_classes"],
                                                pretrained=model_info["pretrained"])
    return model, model_info

def save_checkpoint(model, checkpoint):
    """

    :param model: pytorch model object.
    :param checkpoint: checkpoint path
    :return: None

    """
    target_model = model
    # Check the model is the Dataparallel model or not
    if isinstance(model, (torch.nn.parallel.DataParallel)):
        target_model = model.module
    torch.save(target_model.state_dict(), checkpoint)
    pass

def load_checkpoint(model, checkpoint, perfect_match = False):
    """
    :param model: pytorch model object
    :param checkpoint: checkpoint path
    :return: model.
    """
    checkpoint = torch.load(checkpoint,  map_location="cpu")

    # Check the checkpoint object is Dataparallel module or not
    checkpoint_keys = [key for key in checkpoint.keys()]

    if "module" == checkpoint_keys[0][:6]:
        # If Dataparallel module use,
        # the model architecture and weights are registered as
        # "module" instance of the Dataparalle module.
        print("{} is save as torch.nn.Dataparallel module. ")
        for key in checkpoint_keys:
            checkpoint[key[7:]] = checkpoint.pop(key)

    model.load_state_dict(checkpoint)

    return model