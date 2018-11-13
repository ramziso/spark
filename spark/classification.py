import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
from collections import OrderedDict
import numpy as np
from .pytorch_tools import model_handler, loss_func_handler, layer_manipulation, optimizer_handler, scheduler_handler
from .logger import logger
from .adversarialtraining import VATLoss
from .analyze import ActivationMap
from visdom import Visdom
from .logger.visdom import lineplotstream, plot_images, textstream

import time
from sklearn.metrics import confusion_matrix

def visdom_server():
    import subprocess
    from visdom import Visdom
    visdom_server = subprocess.Popen(["python3","-m", "visdom.server"],
                                     shell=True, stdout= subprocess.PIPE, preexec_fn=os.setsid)
    return visdom_server

def get_instances(cls):
    refs = []
    for ref in cls.__refs__[cls]:
        instance = ref()
        if instance is not None:
            refs.append(ref)
            yield instance

def lists_classes(path):
    return sorted([x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])

class SerializedTrainer():
    def __init__(self, train_folder, test_folder, log_folder, train_transforms = [],
                 test_transforms = [], dataloader_worker = 0, cuda_devices = 0, use_visdom = False):
        self.models = {}
        self.log_folder = log_folder
        self.train_folder = [train_folder]
        self.test_folder = [test_folder]
        self.num_classes = len([folder for folder in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, folder))])
        self.train_transforms  = train_transforms
        self.test_transforms = test_transforms
        self.dataloader_worker = dataloader_worker
        self.class_to_idx = lists_classes(train_folder)
        self.cuda_devices = cuda_devices

        if os.path.exists(self.log_folder) == False:
            os.mkdir(self.log_folder)

        self.use_visdom = use_visdom
        if use_visdom:
            visdom_main = Visdom()
            self.visdom = [visdom_main]

        print("spark SerializedTrainer initialized.")
        print(self)


    def __str__(self):
        return "Classification Serialized Training\n" + \
               "\t(train folder) : \n{} ".format("".join([" "*12 + folder + "\t\t\n" for folder in self.test_folder]))+ \
               "\t(test folder) : \n{}".format("".join([" "*12 + folder + "\t\t\n" for folder in self.test_folder]))+ \
               "\t(num_classes) : {} \n".format(self.num_classes)+ \
               "\t(train_transforms) : \n{} ".format("".join([" "*12 + str(transforms) + "\n" for transforms in self.train_transforms]))+ \
               "\t(test_transforms) : {} \n".format("".join([" "*12 + str(transforms) + "\n"  for transforms in self.test_transforms]))+ \
               "\t(class_to_idx) : {} \n".format(self.class_to_idx)+ \
               "\t(registered models) : {} \n".format(self.models)+ \
               "\t(cuda devices) : {} \n".format(self.cuda_devices)

    def model_summary(self, model_info):
        print_row = ("model_name", "input_size", "pretrained",
                     "train_last_layer", "max_epoch", "batch_size",
                     "lr", "optimizer", "loss_func", "mean", "std",
                     "num_classes", "class_to_idx")
        return "\n\t".join([ "(" + info + ")" + " : "+ str(model_info[info]) for info in print_row])

    def add_model(self, model_name, model_object=None, input_size = None, pretrained = None,
                  train_last_layer = False, max_epoch = 50, batch_size = 16, lr = 0.001, optimizer = "Adam", loss_func = "CrossEntropyLoss",
                  learning_scheduler = None,
                  mean = [0.5,0.5,0.5], std = [0.5, 0.5, 0.5]):
        # Add model registered in pretrained models.
        # Pytorch model object is not defined in this case.
        model_info = OrderedDict({})
        model_info.setdefault("model_name", model_name)
        model_info.setdefault("model_constructer", model_object)
        model_info.setdefault("input_size", input_size)
        model_info.setdefault("pretrained", pretrained)
        model_info.setdefault("train_last_layer", train_last_layer)
        model_info.setdefault("max_epoch", max_epoch)
        model_info.setdefault("batch_size", batch_size)
        model_info.setdefault("lr", lr)
        model_info.setdefault("learning_scheduler", learning_scheduler)
        model_info.setdefault("optimizer", optimizer)
        model_info.setdefault("loss_func", loss_func)
        model_info.setdefault("mean", mean)
        model_info.setdefault("std", std)
        model_info.setdefault("num_classes", self.num_classes)
        model_info.setdefault("class_to_idx", self.class_to_idx)

        # Test model one time. Create model object based on model info
        test_model, _ = model_handler.create_model_object(model_info)
        test_tensor = torch.randn(1, input_size[0],input_size[1],input_size[2])   # [Batch Size, Num_Channel, img_Height, igm_Width]

        if pretrained != None and pretrained == "imagenet":
            try:
                input_size, std, mean = test_model.input_size, test_model.std, test_model.mean
                if model_info["input_size"] != input_size :
                    raise ValueError ("You input wrong input image size for {} imagenet pretrained model setting {}".format(model_name, input_size))
                if model_info["mean"] != mean :
                    raise ValueError ("You input wrong mean value for {} imagenet pretrained model setting {}".format(model_name, mean))
                if model_info["std"] != std :
                    raise ValueError("You input wrong std value for {} imagenet pretrained model setting {}".format(model_name, std))
            except:
                print("Imagenet Pretrain setting for {} not found".format(model_name))

        try:
            test_model.forward(test_tensor)
            print("Model {} is successfully registered to spark".format(model_name))
        except:
            print ("Model {} cannot process sample tensor. Please check the model setting".format(model_name))


        """
        # Adaptation to pretrained models values. (especially you choosed to use "imagenet" pretrained models)
        model_instances = get_instances(test_model)

        test_instances = []
        for instance in model_instances:
            test_instances.append(instance)
        print(test_instances)

        if pretrained != None and pretrained == "imagenet":
            if "input_size" in model_instances:
                if model_info["input_size"] != test_model.input_size:
                    raise ValueError ("Input_size {} must be {} if you use {}'s imagenet pretrained model. ".format(
                        model_info["input_size"], test_model.input_size, model_name))
            if "mean" in model_instances:
                if model_info["mean"] != test_model.mean:
                    raise ValueError("Mean value for normalization {} must be {} if you use {}'s imagenet pretrained model. ".format(
                        model_info["mean"], test_model.mean, model_name))
            if "std" in model_instances:
                if model_info["std"] != test_model.std:
                    raise ValueError("Std value for normalization {} must be {} if you use {}'s imagenet pretrained model. ".format(
                        model_info["std"], test_model.std, model_name))
        """

        self.models.setdefault(model_name, model_info)   # key : model_name ; value : model_info
        print(self.model_summary(model_info))

    def add_train_folder(self, path):
        if self.num_classes != len([folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]):
            raise ValueError("{} contains different number of class to former registered train dataset {}".format(path, self.train_folder))
        else:
            self.train_folder.append(path)
            print

    def add_test_folder(self, path):
        if self.num_classes != len([folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]):
            raise ValueError("{} contains different number of class to former registered test dataset {}".format(path, self.test_folder))
        else:
            self.test_folder.append(path)

    def add_train_transform(self, target_transforms):
        if type(target_transforms) != tuple or type(target_transforms) != list:
            if type(target_transforms) != transforms:
                raise ValueError("target_transforms {} is not torchvision transformation object".format(target_transforms))
            self.train_transforms.append(target_transforms)
        else:
            for target_transform in target_transforms:
                self.target_transforms.append(target_transform)

    def add_test_transform(self, target_transforms):
        if type(target_transforms) != tuple or type(target_transforms) != list:
            if type(target_transforms) != transforms:
                raise ValueError("target_transforms {} is not torchvision transformation object".format(target_transforms))
            self.test_transforms.append(target_transforms)
        else:
            for target_transform in target_transforms:
                self.test_transforms.append(target_transform)

    def benchmark_model(self, model, test_tensor):
        inference_time = []
        model.eval()
        for i in range(50):
            benchmark = time.time()
            model.forward(test_tensor)
            inference_time.append(time.time() - benchmark)
        inference_time = np.array(inference_time)
        inference_avg, inference_std = np.average(inference_time), np.std(inference_time)
        return inference_avg, inference_std

    def benchmark_all(self):
        print("Benchmark for all registered model Starts.")
        benchmark_cpu = {}
        benchmark_gpu = {}
        for model_name in sorted(self.models.keys()):
            model_info = self.models[model_name]
            test_model, _ = model_handler.create_model_object(model_info)
            input_size = model_info["input_size"]
            test_tensor = torch.randn((1, input_size[0], input_size[1], input_size[2]))
            inference_avg, inference_std = self.benchmark_model(test_model, test_tensor)
            print("{} CPU benchmark result : {:.3}+-{:.3}".format(model_name, inference_avg, inference_std))
            benchmark_cpu.setdefault(model_name, [inference_avg, inference_std])
            if torch.cuda.is_available():
                test_model = test_model.cuda()
                test_tensor = test_tensor.cuda()
                gpu_inference_avg, gpu_inference_std = self.benchmark_model(test_model, test_tensor)
                print("{} GPU benchmark result : {:.3}+-{:.3}".format(model_name, gpu_inference_avg, gpu_inference_std))
                benchmark_gpu.setdefault(model_name, [gpu_inference_avg, gpu_inference_std])
            del test_model, test_tensor
        excel_path = os.path.join(self.log_folder, "benchmark_result.xlsx")
        logger.save_excel(excel_path, [("benchmark_cpu", benchmark_cpu),
                                      ("benchmark_gpu", benchmark_gpu)])
        print("Benchmark Finished. Result is saved at {}/benchmark_result.xlsx".format(self.log_folder))

    def augmentation_sample(self, samples_num = 30):
        train_transforms = transforms.Compose(self.train_transforms.copy())
        train_folder = datasets.ImageFolder(self.train_folder[0])

        total_img = []
        for num in range(samples_num):
            total_img.append(np.array(train_transforms(train_folder.__getitem__(1)[0])))

        logger.gridimages(os.path.join(self.log_folder, "Augmentation_result.png"), total_img,
                          cols=5, subtitles=None, title="Data Augmentation result")

        return total_img

    def train_model(self, epoch, train_loader, model, loss_func, optimizer):
        model.train()
        train_correct = 0.0
        train_img = 0.0
        train_loss = 0.0

        for img, label in train_loader:
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()
            img.requires_grad = True
            output = model.forward(img)

            vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
            lds = vat_loss(model, img)
            loss = loss_func(output, label) + 1.0 * lds # VAT Loss training

            loss.backward()
            optimizer.step()
            probability, predict = torch.max(output, 1)

            train_img += label.shape[0]
            train_correct += torch.sum(predict == label).item()
            train_loss += loss.item()
            batch_acc = torch.sum(predict == label).item() / label.shape[0]
            print("EPOCH [{}] TRAIN ACC [{:.5}] LOSS [{:.5}]".format(epoch, batch_acc, loss.item()))
        train_acc = train_correct / train_img
        train_loss = train_loss / train_img
        print("EPOCH [{}] TRAIN RESULT : ACC [{:.5}] LOSS [{:.5}]".format(epoch, train_acc, train_loss))
        return model, train_acc, train_loss

    def __create_dataloader(self, data_folders, data_transforms,  input_size, batch_size, mean, std, shuffle, with_path = False, type = "tensor"):
        final_transforms = transforms.Compose(data_transforms.copy())

        if type == "tensor":
            final_transforms.transforms.append(transforms.Resize((input_size[1], input_size[2])))
            final_transforms.transforms.append(transforms.ToTensor())
            final_transforms.transforms.append(transforms.Normalize(mean, std))

        if with_path == False:
            final_folder = [datasets.ImageFolder(folder, final_transforms) for folder in data_folders]
        else:
            from .dataloader import ImageFolderWithPaths
            final_folder = [ImageFolderWithPaths(folder, final_transforms) for folder in data_folders]
        final_folder = torch.utils.data.ConcatDataset(final_folder)

        if with_path == False:
            final_folder = torch.utils.data.DataLoader(final_folder, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=self.dataloader_worker)
        else:
            # current version of the ImageFolderWithPaths can only be used with bach_size 1.
            final_folder = torch.utils.data.DataLoader(final_folder, batch_size=1,
                                                   shuffle=shuffle, num_workers=self.dataloader_worker)
        return final_folder

    def test_model(self, epoch, test_loader, model, loss_func):
        model.eval()
        test_correct = 0.0
        test_img = 0.0
        test_loss = 0.0
        for img, label in test_loader:
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
            img.requires_grad = False
            output = model.forward(img)
            loss = loss_func(output, label)
            probability, predict = torch.max(output, 1)
            test_img += label.shape[0]
            test_correct += torch.sum(predict == label).item()
            test_loss += loss.item()
            batch_acc = torch.sum(predict == label).item() / label.shape[0]
            print("EPOCH [{}] TEST ACC [{:.5}] LOSS [{:.5}]".format(epoch, batch_acc, loss.item()))
        test_acc = test_correct / test_img
        test_loss = test_loss / test_img
        print("EPOCH [{}] TEST RESULT : ACC [{:.5}] LOSS [{:.5}]".format(epoch, test_acc, test_loss))
        return test_acc, test_loss

    def model_cuda_selecter(self, model):
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).cuda()
        elif torch.cuda.device_count() == 1 :
            model = model.cuda()
        return model

    def tensor_cuda_selecter(self, tensor, device):
        return tensor.to(self.cuda_devices)

    def train_test_model(self, model_info):
        model_name = model_info["model_name"]
        print("[Current model] : " , model_name, "\nModel setting : ", model_info)
        model, model_info = model_handler.create_model_object(model_info)

        model = self.model_cuda_selecter(model)

        # If you want to use pre-trained model, and train the last layer, set the "train_last_layer" as True
        if model_info["train_last_layer"] :
            model = layer_manipulation.train_last_layer(model)

        # create transform object.
        train_loader = self.__create_dataloader(self.train_folder, self.train_transforms,
                                              model_info["input_size"], model_info["batch_size"],
                                              model_info["mean"], model_info["std"], True)
        test_loader = self.__create_dataloader(self.test_folder, self.test_transforms,
                                             model_info["input_size"], model_info["batch_size"],
                                             model_info["mean"], model_info["std"], False)

        loss_func = loss_func_handler.create_loss_func(loss_func_name=model_info["loss_func"])
        optimizer = optimizer_handler.create_optimizer(model, learning_rate=model_info["lr"],
                                                       optimizer_name=model_info["optimizer"])

        epoch_train_acc_list, epoch_train_loss_list = [], []
        epoch_test_acc_list, epoch_test_loss_list = [], []

        # create a several folders into the log folder

        model_main_path = os.path.join(self.log_folder, model_name)
        model_save_path = os.path.join(self.log_folder, model_name, "model_epoch")
        best_model_save_path = os.path.join(self.log_folder, model_name, "best_model")
        if os.path.exists(model_main_path) == False:
            os.mkdir(model_main_path)
        if os.path.exists(model_save_path) == False:
            os.mkdir(model_save_path)
        if os.path.exists(best_model_save_path) == False:
            os.mkdir(best_model_save_path)
        txt_path = os.path.join(model_main_path, model_name + "_train_log.txt")
        model_txt_path = os.path.join(model_main_path, model_name + "_model_architecture.txt")
        train_environment_path = os.path.join(model_main_path, model_name + "_train_environment.txt")
        excel_path = os.path.join(model_main_path, model_name + "_train_result.xlsx")

        result_excel_log = {"train_acc":epoch_train_acc_list, "train_loss":epoch_train_loss_list ,
                            "test_acc":epoch_test_acc_list, "test_loss": epoch_test_loss_list}

        logger.save_txt(model_txt_path, model.__str__())
        logger.save_txt(train_environment_path, train_loader.__str__())
        logger.save_txt(train_environment_path, test_loader.__str__())
        logger.save_txt(train_environment_path, str(model_info))

        best_acc = 0.0
        best_loss = 10000.0

        logger.save_txt(txt_path, "EPOCH, train_acc, train_loss, test_acc, test_loss")

        if self.use_visdom:
            model_visdom = Visdom(env=model_name)
            self.visdom.append(model_visdom)
            model_visdom.text(model.__str__(), opts=dict(title="model architecture"))
            train_environment = model_visdom.text(train_loader.__str__(), opts=dict(title="Training setting"))
            model_visdom.text(test_loader.__str__(), win=train_environment, opts=dict(title="Training setting"), append=True)
            model_visdom.text(str(model_info), win=train_environment, opts=dict(title="Training setting"), append=True)

            model_acc_graph = lineplotstream(model_visdom, "{} Train Test Accuracy".format(model_name))
            model_loss_graph = lineplotstream(model_visdom, "{} Train Test Loss".format(model_name))

        for epoch in range(1,model_info["max_epoch"]+1):
            torch.set_grad_enabled(True)
            model, train_acc, train_loss = self.train_model(epoch, train_loader, model, loss_func, optimizer)
            torch.set_grad_enabled(False)
            test_acc, test_loss = self.test_model(epoch, test_loader, model, loss_func)
            epoch_train_acc_list.append(train_acc)
            epoch_train_loss_list.append(train_loss)
            epoch_test_acc_list.append(test_acc)
            epoch_test_loss_list.append(test_loss)
            text_log = [str(x) for x in [epoch, train_acc, train_loss, test_acc, test_loss]]
            text_log = "\t".join(text_log)
            logger.save_txt(txt_path, text_log)
            #print(result_excel_log)
            logger.save_excel(excel_path, [("{}_result".format(model_name), result_excel_log)])

            if self.use_visdom:
                model_acc_graph.update([train_acc], [epoch], legend="Train")
                model_acc_graph.update([test_acc], [epoch], legend="Test")
                model_loss_graph.update([train_loss], [epoch], legend="Train")
                model_loss_graph.update([test_acc], [epoch], legend="Test")

            model_handler.save_checkpoint(model, os.path.join(model_save_path, "epoch_{}.pth".format(epoch)))

            if best_acc < test_acc:
                model_handler.save_checkpoint(model, os.path.join(best_model_save_path, "best_acc_model.pth".format(epoch)))
                best_acc = test_acc
            if best_loss > test_loss:
                model_handler.save_checkpoint(model, os.path.join(best_model_save_path, "best_loss_model.pth".format(epoch)))
                best_loss = test_loss

        return model, epoch_train_acc_list, epoch_test_loss_list, epoch_test_acc_list, epoch_test_loss_list

    def train_each_models(self):
        augmentated_result = self.augmentation_sample()

        if self.use_visdom:
            main_visdom = self.visdom[0]
            main_visdom.text(str(self), opts=dict(title="Entire training Environment"))
            plot_images(main_visdom, augmentated_result, option=dict(title="Augmentation exemple"))
            class_representative = logger.sample_one_image(self.train_folder[0])
            class_list = sorted(class_representative.keys())
            img_list = [class_representative[key] for key in class_list]
            plot_images(main_visdom, img_list, option=dict(title="Each Classes Image", caption=class_list))


        models_information = {}
        results_epoch_train_acc = {}
        results_epoch_test_acc = {}
        results_epoch_train_loss = {}
        results_epoch_test_loss = {}

        if self.use_visdom:
            all_train_accuracy_plot = lineplotstream(self.visdom[0], "All Model Train Accuracy")
            all_train_loss_plot = lineplotstream(self.visdom[0], "All Model Train Loss")
            all_test_accuracy_plot = lineplotstream(self.visdom[0], "All Model Test Accuracy")
            all_test_loss_plot = lineplotstream(self.visdom[0], "All Model Test Loss")

        for model_name in sorted(self.models.keys()):
            model_info = self.models[model_name]
            model, epoch_train_acc_list, epoch_train_loss_list, epoch_test_acc_list, epoch_test_loss_list = self.train_test_model(model_info)
            models_information.setdefault(model_name, model_info)
            results_epoch_train_acc.setdefault(model_name, epoch_train_acc_list)
            results_epoch_train_loss.setdefault(model_name, epoch_train_loss_list)
            results_epoch_test_acc.setdefault(model_name, epoch_test_acc_list)
            results_epoch_test_loss.setdefault(model_name, epoch_test_loss_list)

            # Logging into Visdom if use_visdom
            if self.use_visdom:
                all_train_accuracy_plot.update(epoch_train_acc_list, np.arange(1, len(epoch_train_acc_list)+1),
                                               model_name)
                all_train_loss_plot.update(epoch_train_loss_list, np.arange(1, len(epoch_train_acc_list) + 1),
                                               model_name)
                all_test_accuracy_plot.update(epoch_test_acc_list, np.arange(1, len(epoch_train_acc_list) + 1),
                                               model_name)
                all_test_loss_plot.update(epoch_test_loss_list, np.arange(1, len(epoch_train_acc_list) + 1),
                                               model_name)

        # Logging all of the informations to excel and graph.
        excel_path = os.path.join(self.log_folder, "summary.xlsx")
        logger.save_excel(excel_path, [("test_acc", results_epoch_test_acc),
                                      ("test_loss", results_epoch_test_loss),
                                      ("train_acc", results_epoch_train_acc),
                                      ("train_loss", results_epoch_train_loss)])
        logger.save_2D_graph(results_epoch_test_acc, "epoch", "test_acc", "test_acc_comparison", os.path.join(self.log_folder, "test_acc_comparison.png"))
        logger.save_2D_graph(results_epoch_test_loss, "epoch", "test_loss", "test_loss_comparison", os.path.join(self.log_folder, "test_loss_comparison.png"))
        logger.save_2D_graph(results_epoch_train_acc, "epoch", "train_acc", "train_acc_comparison", os.path.join(self.log_folder, "train_acc_comparison.png"))
        logger.save_2D_graph(results_epoch_train_loss, "epoch", "train_loss", "train_loss_comparison", os.path.join(self.log_folder, "train_loss_comparison.png"))

    def draw_activation_maps(self, imgs, labels, model, activation_map_save_path, input_size=(3,224,224), mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
        test_transforms = transforms.Compose(self.test_transforms + [transforms.Resize((input_size[1], input_size[2])),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean, std)])
        AM = ActivationMap(model, transforms=test_transforms,  dim=2)
        activation_maps =[]
        for img, label, path in imgs:
            activation_maps.append(AM.draw_activation_map_2d(imgs, labels))
        logger.gridimages(os.path.join(activation_map_save_path, "activation_map.png"), imgs)

    def extract_features_logits(self, model, dataset_loader, logits_save_path):
        init = False
        features_array = None
        logits_array = None
        labels_array = None
        predicts_array = None
        probs_array = None
        path_list = []

        for img, label, path in dataset_loader:
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
            img.requires_grad = False
            label.requires_grad = False

            # check the model is Dataparallel object or not.
            if isinstance(model, (torch.nn.parallel.DataParallel)):
                batch_features = model.module.features(img)
                batch_logits = model.module.logits(batch_features)
            else:
                batch_features = model.features(img)
                batch_logits = model.logits(batch_features)

            batch_features = batch_features.view(batch_features.shape[0], -1)
            label = label.view(label.shape[0], -1)
            _, predict = torch.max(batch_logits, 1)

            batch_prob = F.softmax(batch_logits)

            if init == False:
                features_array = batch_features.detach().cpu().numpy()
                logits_array = batch_logits.detach().cpu().numpy()
                labels_array = label.detach().cpu().numpy()
                predicts_array = predict.detach().cpu().numpy()
                probs_array = batch_prob.detach().cpu().numpy()
                init = True
            else:
                features_array = np.concatenate((features_array, batch_features.detach().cpu().numpy()))
                logits_array = np.concatenate((logits_array, batch_logits.detach().cpu().numpy()))
                labels_array = np.concatenate((labels_array, label.detach().cpu().numpy()))
                predicts_array = np.concatenate((predicts_array, predict.detach().cpu().numpy()))
                probs_array = np.concatenate((probs_array, batch_prob.detach().cpu().numpy()))

            path_list.append(path[0])

        logger.features_to_excel(path_list, features_array, labels_array, logits_save_path)
        logger.logit_to_excel(path_list, logits_array, labels_array, logits_save_path)

        cm = confusion_matrix(predicts_array, labels_array)
        logger.cm_to_excel(cm, self.class_to_idx, logits_save_path)

        # List up the top 5 and top 5 images that most good top1 prediction and worst 5 prediction on every classes.
        # If the Classes number is too many, It will ignore the above 10th classes.
        for idx, top_list, bottom_list, top_list_prob, bottom_list_prob in self.sample_top_bottom_data(path_list, probs_array, labels_array):
            from PIL import Image
            #top_list = utils.make_grid(top_list, nrow=len(top_list))
            #bottom_list = utils.make_grid(bottom_list, nrow=len(bottom_list))

            top_list_img = [np.array(Image.open(img).resize((256,256))) for img in top_list]
            bottom_list_img = [np.array(Image.open(img).resize((256,256))) for img in bottom_list]
            total_img = top_list_img + bottom_list_img

            top_list = [logger.path_leaf(file_path) + "\n"+ "prediction : " + str(prob)[:5] for file_path, prob in zip(top_list, top_list_prob )]
            bottom_list = [logger.path_leaf(file_path) + "\n" + "prediction : " + str(prob)[:5] for file_path, prob in zip(bottom_list, bottom_list_prob)]
            total_list = top_list + bottom_list

            logger.gridimages(os.path.join(logits_save_path, "Validation_result_Images_{}.png".format(idx)), total_img, cols=2, subtitles=total_list, title=idx + "Classification result")

        if len(self.class_to_idx) <= 20:
            logger.plot_confusion_matrix(cm, self.class_to_idx, logits_save_path, normalize=True,
                                         title='Normalized confusion matrix')



    def sample_top_bottom_data(self, paths, logits, label, sample_num = 5):
        import pandas as pd
        paths = pd.DataFrame(paths, index= label.reshape(1,-1)[0].tolist(), columns=["file_path"])
        logits = pd.DataFrame(logits, index= label.reshape(1,-1)[0].tolist(),
                                    columns=[x for x in range(logits.shape[1])])
        all = pd.concat([logits, paths], axis=1)
        for idx in range(len(self.class_to_idx)):
            only_one_class = all.loc[idx, :]
            only_one_class = only_one_class.sort_values(by=idx, axis= 0, ascending=False)
            top_n = only_one_class.iloc[:sample_num, -1]
            top_n_prob = only_one_class.iloc[:sample_num, idx]
            bottom_n = only_one_class.iloc[-sample_num:, -1]
            bottom_n_prob = only_one_class.iloc[-sample_num:, idx]
            yield self.class_to_idx[idx], top_n.tolist(), bottom_n.tolist(), top_n_prob.tolist(), bottom_n_prob.tolist()

    def analyze_models(self):
        # Draw Weight Histogram on Model
        for model_name in self.models.keys():
            model_info = self.models[model_name]
            checkpoint_path = os.path.join(self.log_folder, model_info["model_name"], "best_model",
                                           "best_acc_model.pth")
            model, _ = model_handler.create_model_object(model_info)
            model_handler.load_checkpoint(model, checkpoint_path)
            logger.draw_weight_histogram(model, os.path.join(self.log_folder, model_info["model_name"]))

    def features_logits_from_each_models(self):
        for model_name in self.models.keys():
            model_info = self.models[model_name]
            checkpoint_path = os.path.join(self.log_folder, model_info["model_name"],
                                           "best_model", "best_acc_model.pth")
            model, _ = model_handler.create_model_object(model_info)
            model_handler.load_checkpoint(model, checkpoint_path)

            model = self.model_cuda_selecter(model)
            model.eval()

            train_logits_save_path = os.path.join(self.log_folder, model_info["model_name"], "logits_train")
            test_logits_save_path = os.path.join(self.log_folder, model_info["model_name"], "logits_test")

            os.makedirs(train_logits_save_path, exist_ok=True)
            os.makedirs(test_logits_save_path, exist_ok=True)

            train_loader = self.__create_dataloader(self.train_folder,
                                                    self.test_transforms,
                                                    model_info["input_size"],
                                                    model_info["batch_size"],
                                                    model_info["mean"],
                                                    model_info["std"],
                                                    shuffle=False, with_path=True)
            test_loader = self.__create_dataloader(self.test_folder,
                                                   self.test_transforms,
                                                   model_info["input_size"],
                                                   model_info["batch_size"],
                                                   model_info["mean"],
                                                   model_info["std"],
                                                   shuffle=False, with_path=True)

            self.extract_features_logits(model, train_loader,
                                         logits_save_path=train_logits_save_path)
            self.extract_features_logits(model, test_loader,
                                         logits_save_path=test_logits_save_path)
