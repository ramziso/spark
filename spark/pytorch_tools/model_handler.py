import torch
#import architectures

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