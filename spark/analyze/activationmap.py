import torch
import numpy as np

__all__ = "ActivationMap"

def __check_architecture(model):
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

class ActivationMap():
    def __init__(self, model, transforms, input_size = (3,224,224), dim=2):
        self.dimension = dim
        self.input_size = input_size
        self.target_layer = __check_architecture(model)
        if self.target_layer != None:
            self.target_layer.register_forward_hook(self.get_activation_state())
        else:
            print("model {} architecture cannot adapted to the activation map.")
            return False
            raise Exception ("Activation Map error : Only model that have AvgPooling layer before fc canbe adapt to Activation Map")
        self.features_blob = []

    def get_activation_state(self, input, output):
        return self.features_blob.append(output.detach().data.cpu().numpy())

    def returnCAM2d(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to identical to input_img_size
        bz, nc, h, w = feature_conv.shape
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            yield cam_img

    def draw_activation_map_2d(self, imgs, labels, model, transforms):
        model.eval()
        params = list(model.parameters())
        weight_softmax = np.sqeeze(params[-2].detach().data.numpy())
        imgs = transforms(imgs)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        predict = model.forward(imgs)
        CAM = self.returnCAM2d(self.features_blob. weight_softmax, labels)
        return CAM