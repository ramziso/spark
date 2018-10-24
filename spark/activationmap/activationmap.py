import torch
import numpy as np
from PIL import Image

"__all__" = ""


def check_architecture(model):

    layers = reversed([layer for layer in model.children()])[:2]

    for layer in layers:
        if isinstance(layer, (torch.nn.AdaptiveAvgPool1d, torch.nn.AvgPool1d)):
            return layer
        elif isinstance(layer, (torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d)):
            return layer
        elif isinstance(layer, (torch.nn.AdaptiveAvgPool3d, torch.nn.AvgPool3d)):
            return layer
        else:
            return None

def draw_activation_map_1d(vector, model,transforms):

    pass


def draw_activation_map_2d(imgs, labels, model, transforms):
    features_blob = []
    def get_activation_state(self, input, output):
        return features_blob.append(output.detach().data.cpu().numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to identical to input_img_size
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            yield cam_img

    # check the model is adapted to Activation maps by checking the second-last layer type.
    layer = check_architecture(model)

    if layer != None:
        layer.register_forward_hook(get_activation_state())
    else:
        print("model {} architecture cannot adapted to the activation map.")
        pass

    model.eval()
    params = list(model.parameters())
    weight_softmax = np.sqeeze(params[-2].detach().data.numpy())

    imgs = transforms(imgs)
    if torch.cuda.is_available():
        img = img.cuda()
    predict = model.forward(img)
    returnCAM


def draw_activation_map_3d(movie, model, transforms):

    pass