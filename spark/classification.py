import torch
import torch.nn as nn
from torchvision import datasets, utils, transforms
import os
import numpy as np
from .pytorch_tools import model_handler , loss_func_handler, layer_manipulation, optimizer_handler
from .pytorch_tools import logger
import time
import codecs
from sklearn.metrics import confusion_matrix

def get_instances(cls):
    refs = []
    for ref in cls.__refs__[cls]:
        instance = ref()
        if instance is not None:
            refs.append(ref)
            yield instance
    # print(len(refs))
    cls.__refs__[cls] = refs

def lists_classes(path):
    return sorted([x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])

class SerializedTrainer():
    def __init__(self, train_folder, test_folder, log_folder, train_transforms = [],
                 test_transforms = [], dataloader_worker = 0, cuda_devices = 0):
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

    def add_model(self, model_name, model_object=None, input_size = None, pretrained = None,
                  train_last_layer = False, max_epoch = 50, batch_size = 16, lr = 0.001, optimizer = "Adam", loss_func = "CrossEntropyLoss",
                  mean = [0.5,0.5,0.5], std = [0.5, 0.5, 0.5]):
        # Add model registered in pretrained models.
        # Pytorch model object is not defined in this case.
        model_info = {}
        model_info.setdefault("model_name", model_name)
        model_info.setdefault("model_constructer", model_object)
        model_info.setdefault("input_size", input_size)
        model_info.setdefault("pretrained", pretrained)
        model_info.setdefault("train_last_layer", train_last_layer)
        model_info.setdefault("max_epoch", max_epoch)
        model_info.setdefault("batch_size", batch_size)
        model_info.setdefault("lr", lr)
        model_info.setdefault("optimizer", optimizer)
        model_info.setdefault("loss_func", loss_func)
        model_info.setdefault("model_object", None)
        model_info.setdefault("mean", mean)
        model_info.setdefault("std", std)
        model_info.setdefault("num_classes", self.num_classes)
        model_info.setdefault("class_to_idx", self.class_to_idx)
        # Test model one time.
        test_model, _ = model_handler.create_model_object(model_info)
        test_tensor = torch.randn(1, input_size[0],input_size[1],input_size[2])
        test_model.forward(test_tensor)

        # Adaptation to pretrained models values. (especially you choosed to use "imagenet" pretrained models
        model_instances = get_instances(test_model)
        if pretrained != None and pretrained == "imagenet":
            if "input_size" in model_instances:
                if model_info["input_size"] != test_model.input_size:
                    raise ValueError ("Input_size {} is must be {} if you use {}'s imagenet pretrained model. ".format(
                        model_info["input_size"], test_model.input_size, model_name))
            if "mean" in model_instances:
                if model_info["input_size"] != test_model.mean:
                    raise ValueError("Mean value for normalization {} is must be {} if you use {}'s imagenet pretrained model. ".format(
                        model_info["mean"], test_model.mean, model_name))
            if "std" in model_instances:
                if model_info["input_size"] != test_model.std:
                    raise ValueError("Std value for normalization {} is must be {} if you use {}'s imagenet pretrained model. ".format(
                        model_info["std"], test_model.std, model_name))

        self.models.setdefault(model_name, model_info)

    def add_train_folder(self, path):
        if self.num_classes != len([folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]):
            raise ValueError("{} contains different number of class to former registered train dataset {}".format(path, self.train_folder))
        else:
            self.train_folder.append(path)

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

    def add_test_transform(self, transforms):
        self.test_transforms.append(transforms)

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
        print("Benchmark for all registered model Start.")
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
        logger.log_excel(excel_path, [("benchmark_cpu", benchmark_cpu),
                                      ("benchmark_gpu", benchmark_gpu)])
        print("Benchmark Finished. Result is saved at {}/benchmark_result.xlsx".format(self.log_folder))

    def train_model(self, epoch, train_loader, model, loss_func, optimizer):
        model.train()
        train_correct = 0.0
        train_img = 0.0
        train_loss = 0.0
        for img, label in train_loader:
            img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()
            img.requires_grad = True
            output = model.forward(img)
            loss = loss_func(output, label)
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

    def __create_dataloader(self, data_folders, data_transforms,  input_size, batch_size, mean, std, shuffle):
        final_transforms = transforms.Compose(data_transforms.copy())
        final_transforms.transforms.append(transforms.Resize((input_size[1], input_size[2])))
        final_transforms.transforms.append(transforms.ToTensor())
        final_transforms.transforms.append(transforms.Normalize(mean, std))

        final_folder = [datasets.ImageFolder(folder, final_transforms) for folder in data_folders]
        final_folder = torch.utils.data.ConcatDataset(final_folder)
        final_folder = torch.utils.data.DataLoader(final_folder, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=self.dataloader_worker)
        return final_folder

    def test_model(self, epoch, test_loader, model, loss_func):
        model.eval()
        test_correct = 0.0
        test_img = 0.0
        test_loss = 0.0
        for img, label in test_loader:
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

    def __model_cuda_selecter(self, model):
        if torch.cuda.device_count() > 1:
            model = model.cuda()
            #model = nn.DataParallel(model).cuda()
        elif torch.cuda.device_count() == 1 :
            model = model.cuda()
        return model

    def __tensor_cuda_selecter(self, tensor, device):
        return tensor.to(device)

    def train_test_model(self, model_info):
        model_name = model_info["model_name"]
        print("Current model : " , model_name, "\nModel setting : ", model_info)
        model, model_info = model_handler.create_model_object(model_info)

        model = self.__model_cuda_selecter(model)

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

        logger.log_txt(model_txt_path, model.__str__())
        logger.log_txt(train_environment_path, train_loader.__str__())
        logger.log_txt(train_environment_path, test_loader.__str__())
        logger.log_txt(train_environment_path, str(model_info))

        best_acc = 0.0
        best_loss = 10000.0

        logger.log_txt(txt_path, "EPOCH, train_acc, train_loss, test_acc, test_loss")

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
            logger.log_txt(txt_path, text_log)
            print(result_excel_log)
            logger.log_excel(excel_path, [("{}_result".format(model_name),result_excel_log)])

            torch.save(model.state_dict(), os.path.join(model_save_path, "epoch_{}.pth".format(epoch)))

            if best_acc < test_acc:
                torch.save(model.state_dict(), os.path.join(best_model_save_path, "best_acc_model.pth".format(epoch)))
                best_acc = test_acc
            if best_loss > test_loss:
                torch.save(model.state_dict(), os.path.join(best_model_save_path, "best_loss_model.pth".format(epoch)))
                best_loss = test_loss

        return model, epoch_train_acc_list, epoch_test_loss_list, epoch_test_acc_list, epoch_test_loss_list

    def train_each_models(self):
        models_information = {}
        results_epoch_train_acc = {}
        results_epoch_test_acc = {}
        results_epoch_train_loss = {}
        results_epoch_test_loss = {}
        for model_name in sorted(self.models.keys()):
            model_info = self.models[model_name]
            model, epoch_train_acc_list, epoch_train_loss_list, epoch_test_acc_list, epoch_test_loss_list = self.train_test_model(model_info)
            models_information.setdefault(model_name, model_info)
            results_epoch_train_acc.setdefault(model_name, epoch_train_acc_list)
            results_epoch_train_loss.setdefault(model_name, epoch_train_loss_list)
            results_epoch_test_acc.setdefault(model_name, epoch_test_acc_list)
            results_epoch_test_loss.setdefault(model_name, epoch_test_loss_list)

        # Logging all of the informations to excel and graph.
        excel_path = os.path.join(self.log_folder, "summary.xlsx")
        logger.log_excel(excel_path, [("test_acc", results_epoch_test_acc),
                                      ("test_loss", results_epoch_test_loss),
                                      ("train_acc", results_epoch_train_acc),
                                      ("train_loss", results_epoch_train_loss)])
        logger.plot_graph_save(os.path.join(self.log_folder, "test_acc_comparison.png"),
                               "epoch", "test_acc", "test_acc_comparison", results_epoch_test_acc)
        logger.plot_graph_save(os.path.join(self.log_folder, "test_loss_comparison.png"),
                               "epoch", "test_loss", "test_loss_comparison", results_epoch_test_loss)
        logger.plot_graph_save(os.path.join(self.log_folder, "train_acc_comparison.png"),
                               "epoch", "train_acc", "train_acc_comparison", results_epoch_train_acc)
        logger.plot_graph_save(os.path.join(self.log_folder, "train_loss_comparison.png"),
                               "epoch", "train_loss", "train_loss_comparison", results_epoch_train_loss)

    def extract_features_logits(self, model, dataset_loader, logits_save_path):
        init = False
        features_array = None
        logits_array = None
        labels_array = None

        for img, label in dataset_loader:
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
            img.requires_grad = False
            label.requires_grad = False
            batch_features = model.features(img)
            batch_logits = model.logits(batch_features)

            batch_features = batch_features.view(batch_features.shape[0], -1)
            label = label.view(label.shape[0], -1)

            if init == False:
                features_array = batch_features.detach().cpu().numpy()
                logits_array = batch_logits.detach().cpu().numpy()
                labels_array = label.detach().cpu().numpy()
                init = True
            else:
                features_array = np.concatenate((features_array, batch_features.detach().cpu().numpy()))
                logits_array = np.concatenate((logits_array, batch_logits.detach().cpu().numpy()))
                labels_array = np.concatenate((labels_array, label.detach().cpu().numpy()))

        logger.features_to_excel(features_array, labels_array, logits_save_path)
        logger.logit_to_excel(logits_array, labels_array, logits_save_path)

        cm = confusion_matrix(np.argmax(logits_array, axis=1), labels_array)
        logger.cm_to_excel(cm, self.class_to_idx ,logits_save_path)

        if len(self.class_to_idx) <= 20:
            logger.plot_confusion_matrix(cm, self.class_to_idx, logits_save_path, normalize=True,
                                      title='Normalized confusion matrix')

    def features_logits_from_each_models(self):
        for model_name in self.models.keys():
            model_info = self.models[model_name]
            checkpoint_path = os.path.join(self.log_folder, model_info["model_name"], "best_model",
                                            "best_acc_model.pth")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model, _ = model_handler.create_model_object(model_info)
            model.load_state_dict(checkpoint)
            model = self.__model_cuda_selecter(model)
            model.eval()

            train_logits_save_path = os.path.join(self.log_folder, model_info["model_name"], "logits_train")
            test_logits_save_path = os.path.join(self.log_folder, model_info["model_name"], "logits_test")

            os.makedirs(train_logits_save_path, exist_ok = True)
            os.makedirs(test_logits_save_path, exist_ok = True)

            train_loader = self.__create_dataloader(self.train_folder, self.test_transforms, model_info["input_size"],
                                                      model_info["batch_size"], model_info["mean"], model_info["std"], shuffle=False)
            test_loader = self.__create_dataloader(self.test_folder, self.test_transforms, model_info["input_size"],
                                                    model_info["batch_size"], model_info["mean"], model_info["std"], shuffle=False)

            self.extract_features_logits(model, train_loader, logits_save_path=train_logits_save_path)
            self.extract_features_logits(model, test_loader, logits_save_path=test_logits_save_path)