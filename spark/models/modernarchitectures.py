import torch
import torch.nn as nn
import torch.nn.functional as F
import types

try:
    import pretrainedmodels as models
except:
    "Error! There is no pretrainedmodels in your python environment."

__all__ = ["alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn",
           "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
           "densenet121", "densenet161", "densenet169", "densenet201",
           "inceptionv3", "inceptionv4", "inceptionresnetv2",
           "pnasnet5large", "polynet", "nasnetamobile", "nasnetalarge"
           ]

pretrained_setting = 0

def add_instances_to_torchvisionmodel(model):
    model.input_space = "RGB"
    model.input_size = [3, 224, 224]
    model.mean = [0.485, 0.456, 0.406]
    model.std = [0.229, 0.224, 0.225]
    model.input_range = [0, 1]
    return model

######################## Alexnet #####################

def alexnet(input_size = (3,224,224), num_classes = 1000, pretrained = None):
    model = models.alexnet(num_classes = num_classes, pretrained = None)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size = (11,11), stride = (4,4), padding=(2,2))
        model.input_size = input_size
    # calc "linear0" "linear1" input size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    input_tensor_size = x.view(x.size(0), -1).shape[1]

    # embedding sizes adaptively changed for different input shape.
    if input_tensor_size <= 4096 :
        out_feature_size = input_tensor_size
    else:
        out_feature_size = 4096
    model.linear0 = nn.Linear(in_features=input_tensor_size, out_features= out_feature_size, bias=True)
    model.linear1 = nn.Linear(in_features=out_feature_size, out_features = out_feature_size, bias = True)
    model.last_linear = nn.Linear(in_features=out_feature_size, out_features = num_classes, bias = True)

    print(model.linear0)
    print(model.linear1)
    del model.features
    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), -1)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        return x
    model.features = types.MethodType(features, model)

    return model

######################### DenseNet #########################

def densenet121(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.densenet121(num_classes = num_classes, pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.features[0] = nn.Conv2d(input_size[0], 64, kernel_size = (7,7), stride = (2,2), padding=(3,3), bias=False)
        model.input_size = input_size

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    #print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size = avg_pool2d_kernel_size, stride = 1)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    #del model.logits
    #del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def densenet161(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.densenet161(num_classes = num_classes, pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.features[0] = nn.Conv2d(input_size[0], 96, kernel_size = (7,7), stride = (2,2), padding=(3,3), bias=False)
        model.input_size = input_size

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    #print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size = avg_pool2d_kernel_size, stride = 1)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    #del model.logits
    #del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def densenet169(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.densenet169(num_classes = num_classes, pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.features[0] = nn.Conv2d(input_size[0], 64, kernel_size = (7,7), stride = (2,2), padding=(3,3), bias=False)
        model.input_size = input_size

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    #print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size = avg_pool2d_kernel_size, stride = 1)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    #del model.logits
    #del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def densenet201(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.densenet201(num_classes = num_classes, pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.features[0] = nn.Conv2d(input_size[0], 64, kernel_size = (7,7), stride = (2,2), padding=(3,3), bias=False)
        model.input_size = input_size

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    #print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size = avg_pool2d_kernel_size, stride = 1)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    #del model.logits
    #del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

######################### Resnet #########################

def resnet18(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.resnet18(pretrained=pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.conv1 = nn.Conv2d(input_size[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.input_size = input_size

    del model.fc
    del model.avgpool

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    # print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size=avg_pool2d_kernel_size)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    ##del model.logits
    ##del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet34(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.resnet34(pretrained=pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.conv1 = nn.Conv2d(input_size[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.input_size = input_size

    del model.fc
    del model.avgpool

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    # print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size=avg_pool2d_kernel_size)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    #del model.logits
    #del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet50(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.resnet50(pretrained=pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.conv1 = nn.Conv2d(input_size[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.input_size = input_size

    del model.fc
    del model.avgpool

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    # print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size=avg_pool2d_kernel_size)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    #del model.logits
    #del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet101(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.resnet101(pretrained=pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.conv1 = nn.Conv2d(input_size[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.input_size = input_size

    del model.fc
    del model.avgpool

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    # print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size=avg_pool2d_kernel_size)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    #del model.logits
    #del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet152(input_size=(3,224,224), num_classes=1000, pretrained=None):
    model = models.resnet152(pretrained=pretrained)
    model = add_instances_to_torchvisionmodel(model)
    # Change the First Convol2D layer into new input shape
    if input_size != (3,224,224):
        model.conv1 = nn.Conv2d(input_size[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.input_size = input_size

    del model.fc
    del model.avgpool

    # calc kernel_size on new_avgpool2d layer
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    features = model.features(test_tensor)
    # print(features, features.shape[2], features.shape[3])
    avg_pool2d_kernel_size = (features.shape[2], features.shape[3])

    # calc last linear size
    x = F.avg_pool2d(features, kernel_size=avg_pool2d_kernel_size)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x, out_features=num_classes)

    #del model.logits
    #del model.forward
    def logits(self, features):
        x = F.relu(features, inplace=False)
        x = F.avg_pool2d(x, kernel_size=avg_pool2d_kernel_size, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

####################### VGGs ###############################

def vgg11(input_size = (3,224,224), num_classes=1000, pretrained=None):
    model = models.vgg11(pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.input_size = input_size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    second_out_features = 4096
    if x.shape[1] < 4096:
        second_out_features = x.shape[1]
    model.linear0 = nn.Linear(in_features=x.shape[1], out_features=second_out_features, bias= True)
    model.linear1 = nn.Linear(in_features=second_out_features, out_features=second_out_features, bias= True)
    model.last_linear = nn.Linear(in_features=second_out_features, out_features=num_classes, bias= True)
    return model


def vgg11_bn(input_size = (3,224,224), num_classes=1000, pretrained=None):
    model = models.vgg11_bn(pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.input_size = input_size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    second_out_features = 4096
    if x.shape[1] < 4096:
        second_out_features = x.shape[1]
    model.linear0 = nn.Linear(in_features=x.shape[1], out_features=second_out_features, bias= True)
    model.linear1 = nn.Linear(in_features=second_out_features, out_features=second_out_features, bias= True)
    model.last_linear = nn.Linear(in_features=second_out_features, out_features=num_classes, bias= True)
    return model


def vgg13(input_size = (3,224,224), num_classes=1000, pretrained=None):
    model = models.vgg13(pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.input_size = input_size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    second_out_features = 4096
    if x.shape[1] < 4096:
        second_out_features = x.shape[1]
    model.linear0 = nn.Linear(in_features=x.shape[1], out_features=second_out_features, bias= True)
    model.linear1 = nn.Linear(in_features=second_out_features, out_features=second_out_features, bias= True)
    model.last_linear = nn.Linear(in_features=second_out_features, out_features=num_classes, bias= True)
    return model


def vgg13_bn(input_size = (3,224,224), num_classes=1000, pretrained=None):
    model = models.vgg13_bn(pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.input_size = input_size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    second_out_features = 4096
    if x.shape[1] < 4096:
        second_out_features = x.shape[1]
    model.linear0 = nn.Linear(in_features=x.shape[1], out_features=second_out_features, bias= True)
    model.linear1 = nn.Linear(in_features=second_out_features, out_features=second_out_features, bias= True)
    model.last_linear = nn.Linear(in_features=second_out_features, out_features=num_classes, bias= True)
    return model


def vgg16(input_size = (3,224,224), num_classes=1000, pretrained=None):
    model = models.vgg16(pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.input_size = input_size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    second_out_features = 4096
    if x.shape[1] < 4096:
        second_out_features = x.shape[1]
    model.linear0 = nn.Linear(in_features=x.shape[1], out_features=second_out_features, bias= True)
    model.linear1 = nn.Linear(in_features=second_out_features, out_features=second_out_features, bias= True)
    model.last_linear = nn.Linear(in_features=second_out_features, out_features=num_classes, bias= True)
    return model


def vgg16_bn(input_size = (3,224,224), num_classes=1000, pretrained=None):
    model = models.vgg16_bn(pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.input_size = input_size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    second_out_features = 4096
    if x.shape[1] < 4096:
        second_out_features = x.shape[1]
    model.linear0 = nn.Linear(in_features=x.shape[1], out_features=second_out_features, bias= True)
    model.linear1 = nn.Linear(in_features=second_out_features, out_features=second_out_features, bias= True)
    model.last_linear = nn.Linear(in_features=second_out_features, out_features=num_classes, bias= True)
    return model


def vgg19_bn(input_size = (3,224,224), num_classes=1000, pretrained=None):
    model = models.vgg19_bn(pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.input_size = input_size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    second_out_features = 4096
    if x.shape[1] < 4096:
        second_out_features = x.shape[1]
    model.linear0 = nn.Linear(in_features=x.shape[1], out_features=second_out_features, bias= True)
    model.linear1 = nn.Linear(in_features=second_out_features, out_features=second_out_features, bias= True)
    model.last_linear = nn.Linear(in_features=second_out_features, out_features=num_classes, bias= True)
    return model


def vgg19_bn(input_size = (3,224,224), num_classes=1000, pretrained=None):
    model = models.vgg19_bn(pretrained = pretrained)
    model = add_instances_to_torchvisionmodel(model)
    if input_size != (3,224,224):
        model._features[0] = nn.Conv2d(input_size[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.input_size = input_size
    test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
    x = model._features(test_tensor)
    x = x.view(x.size(0), -1)
    second_out_features = 4096
    if x.shape[1] < 4096:
        second_out_features = x.shape[1]
    model.linear0 = nn.Linear(in_features=x.shape[1], out_features=second_out_features, bias= True)
    model.linear1 = nn.Linear(in_features=second_out_features, out_features=second_out_features, bias= True)
    model.last_linear = nn.Linear(in_features=second_out_features, out_features=num_classes, bias= True)
    return model

################# Modern CNN architecture ###############

# TODO : inceptionv3
def inceptionv3(input_size=(299,299,3), num_classes=1000, pretrained=None):
    model = models.inceptionv3(num_classes= 1000, pretrained = None)
    if input_size != (299,299,3):
        model.features[0].conv = nn.Conv2d(input_size[0], 32, kernel_size=3, stride=2, bias = False)
    pass


def inceptionv4(input_size=(3, 299,299), num_classes=1000, pretrained=None):
    # Minimum Input size = (96,96) one is must above than 97
    model = models.inceptionv4(num_classes= 1000, pretrained = pretrained)
    if input_size != (3, 299,299):
        model.features[0].conv = nn.Conv2d(input_size[0], 32, kernel_size=3, stride=2, bias = False)
        model.input_size = input_size
    # calculate last avgpool_1a kernel size
    test_tensor = torch.randn(1,input_size[0], input_size[1], input_size[2])
    x = model.features(test_tensor)
    print(x.shape)
    model.avg_pool = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]), padding=0)
    x = model.avg_pool(x)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x , out_features=num_classes, bias= True)
    return model

#model = inceptionv4((160,128,4))

def inceptionresnetv2(input_size=(3,299,299), num_classes=1000, pretrained=None):
    # Minimum Input size = (96,96) one is must above than 97
    model = models.inceptionresnetv2(num_classes= 1000, pretrained = pretrained)
    if input_size != (3,299,299):
        model.conv2d_1a.conv = nn.Conv2d(input_size[0], 32, kernel_size= (3,3), stride = (2,2), bias=False)
        model.input_size = input_size
    # calculate last avgpool_1a kernel size
    test_tensor = torch.randn(1,input_size[0], input_size[1], input_size[2])
    x = model.conv2d_1a(test_tensor)
    x = model.conv2d_2a(x)
    x = model.conv2d_2b(x)
    x = model.maxpool_3a(x)
    x = model.conv2d_3b(x)
    x = model.conv2d_4a(x)
    x = model.maxpool_5a(x)
    x = model.mixed_5b(x)
    x = model.repeat(x)
    x = model.mixed_6a(x)
    x = model.repeat_1(x)
    x = model.mixed_7a(x)
    x = model.repeat_2(x)
    x = model.block8(x)
    x = model.conv2d_7b(x)
    #print(x.shape)

    model.avgpool_1a = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]), padding=0)
    x = model.avgpool_1a(x)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x , out_features=num_classes, bias= True)
    return model


def nasnetalarge(input_size=(3,331,331), num_classes=1000, pretrained=None):
    # Minimum Input size = (96,96) one is must above than 97
    model = models.nasnetalarge(num_classes= 1000, pretrained = pretrained)
    if input_size != (3,331,331):
        model.conv0.conv = nn.Conv2d(in_channels=input_size[0], out_channels=96, kernel_size=3, padding=0, stride=2,
                                                bias=False)
        model.input_size = input_size
    # calculate last avgpool_1a kernel size
    test_tensor = torch.randn(1,input_size[0], input_size[1], input_size[2])
    x_conv0 = model.conv0(test_tensor)
    x_stem_0 = model.cell_stem_0(x_conv0)
    x_stem_1 = model.cell_stem_1(x_conv0, x_stem_0)

    x_cell_0 = model.cell_0(x_stem_1, x_stem_0)
    x_cell_1 = model.cell_1(x_cell_0, x_stem_1)
    x_cell_2 = model.cell_2(x_cell_1, x_cell_0)
    x_cell_3 = model.cell_3(x_cell_2, x_cell_1)
    x_cell_4 = model.cell_4(x_cell_3, x_cell_2)
    x_cell_5 = model.cell_5(x_cell_4, x_cell_3)

    x_reduction_cell_0 = model.reduction_cell_0(x_cell_5, x_cell_4)

    x_cell_6 = model.cell_6(x_reduction_cell_0, x_cell_4)
    x_cell_7 = model.cell_7(x_cell_6, x_reduction_cell_0)
    x_cell_8 = model.cell_8(x_cell_7, x_cell_6)
    x_cell_9 = model.cell_9(x_cell_8, x_cell_7)
    x_cell_10 = model.cell_10(x_cell_9, x_cell_8)
    x_cell_11 = model.cell_11(x_cell_10, x_cell_9)

    x_reduction_cell_1 = model.reduction_cell_1(x_cell_11, x_cell_10)

    x_cell_12 = model.cell_12(x_reduction_cell_1, x_cell_10)
    x_cell_13 = model.cell_13(x_cell_12, x_reduction_cell_1)
    x_cell_14 = model.cell_14(x_cell_13, x_cell_12)
    x_cell_15 = model.cell_15(x_cell_14, x_cell_13)
    x_cell_16 = model.cell_16(x_cell_15, x_cell_14)
    x_cell_17 = model.cell_17(x_cell_16, x_cell_15)
    x = model.relu(x_cell_17)
    model.avg_pool = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]))
    x = model.avg_pool(x)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x , out_features=num_classes, bias= True)
    return model


def nasnetamobile(input_size=(3,224,224), num_classes=1001, pretrained=None):
    # Minimum Input size = (96,96) one is must above than 97
    model = models.nasnetamobile(num_classes= 1000, pretrained = pretrained)
    model.conv0.conv = nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=3, padding=0, stride=2,
                                                bias=False)
    # calculate last avgpool_1a kernel size
    test_tensor = torch.randn(1,input_size[0], input_size[1], input_size[2])

    x_conv0 = model.conv0(test_tensor)
    x_stem_0 = model.cell_stem_0(x_conv0)
    x_stem_1 = model.cell_stem_1(x_conv0, x_stem_0)

    x_cell_0 = model.cell_0(x_stem_1, x_stem_0)
    x_cell_1 = model.cell_1(x_cell_0, x_stem_1)
    x_cell_2 = model.cell_2(x_cell_1, x_cell_0)
    x_cell_3 = model.cell_3(x_cell_2, x_cell_1)

    x_reduction_cell_0 = model.reduction_cell_0(x_cell_3, x_cell_2)

    x_cell_6 = model.cell_6(x_reduction_cell_0, x_cell_3)
    x_cell_7 = model.cell_7(x_cell_6, x_reduction_cell_0)
    x_cell_8 = model.cell_8(x_cell_7, x_cell_6)
    x_cell_9 = model.cell_9(x_cell_8, x_cell_7)

    x_reduction_cell_1 = model.reduction_cell_1(x_cell_9, x_cell_8)

    x_cell_12 = model.cell_12(x_reduction_cell_1, x_cell_9)
    x_cell_13 = model.cell_13(x_cell_12, x_reduction_cell_1)
    x_cell_14 = model.cell_14(x_cell_13, x_cell_12)
    x_cell_15 = model.cell_15(x_cell_14, x_cell_13)
    x = model.relu(x_cell_15)
    model.avg_pool = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]))
    x = model.avg_pool(x)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x , out_features=num_classes, bias= True)
    return model


def pnasnet5large(input_size=(3,224,224), num_classes=1001, pretrained=None):
    # Minimum Input size = (96,96) one is must above than 97
    model = models.pnasnet5large(num_classes= 1000, pretrained = pretrained)
    if input_size != (3,331,331):
        model.conv_0.conv = nn.Conv2d(in_channels=input_size[0], out_channels=96, kernel_size=3, stride=2,
                                                bias=False)
        model.input_size = input_size
    # calculate last avgpool_1a kernel size
    test_tensor = torch.randn(1,input_size[0], input_size[1], input_size[2])

    x_conv_0 = model.conv_0(test_tensor)
    x_stem_0 = model.cell_stem_0(x_conv_0)
    x_stem_1 = model.cell_stem_1(x_conv_0, x_stem_0)
    x_cell_0 = model.cell_0(x_stem_0, x_stem_1)
    x_cell_1 = model.cell_1(x_stem_1, x_cell_0)
    x_cell_2 = model.cell_2(x_cell_0, x_cell_1)
    x_cell_3 = model.cell_3(x_cell_1, x_cell_2)
    x_cell_4 = model.cell_4(x_cell_2, x_cell_3)
    x_cell_5 = model.cell_5(x_cell_3, x_cell_4)
    x_cell_6 = model.cell_6(x_cell_4, x_cell_5)
    x_cell_7 = model.cell_7(x_cell_5, x_cell_6)
    x_cell_8 = model.cell_8(x_cell_6, x_cell_7)
    x_cell_9 = model.cell_9(x_cell_7, x_cell_8)
    x_cell_10 = model.cell_10(x_cell_8, x_cell_9)
    x_cell_11 = model.cell_11(x_cell_9, x_cell_10)
    x = model.relu(x_cell_11)
    model.avg_pool = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]))
    x = model.avg_pool(x)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x , out_features=num_classes, bias= True)
    return model


def polynet(input_size=(3,331,331), num_classes=1001, pretrained=None):
    # Minimum Input size = (96,96) one is must above than 97
    model = models.polynet(num_classes= 1000, pretrained = pretrained)
    if input_size != (3,331,331):
        model.stem.conv1[0].conv = nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=(3,3), stride=(2,2),
                                                    bias=False)
        model.input_size = input_size
    # calculate last avgpool_1a kernel size
    test_tensor = torch.randn(1,input_size[0], input_size[1], input_size[2])
    x = model.stem(test_tensor)
    x = model.stage_a(x)
    x = model.reduction_a(x)
    x = model.stage_b(x)
    x = model.reduction_b(x)
    x = model.stage_c(x)
    model.avg_pool = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]))
    x = model.avg_pool(x)
    x = x.view(x.size(0), -1).shape[1]
    model.last_linear = nn.Linear(in_features=x , out_features=num_classes, bias= True)
    return model

"""
size = range (256,500,64)
shapes = [2,3,4]
for size_x in size:
    for size_y in size:
        for shape in shapes:
            input_shape = (size_x, size_y, shape)
            test = torch.randn((1,shape, size_x, size_y))
            print("for input_shape , test start", input_shape)
            model = inceptionresnetv2(input_shape)
            model(test)
            model = nasnetalarge(input_shape)
            model(test)
            model = nasnetamobile(input_shape)
            model(test)
            model = pnasnet5large(input_shape)
            model(test)
            model = polynet(input_shape)
            model(test)
"""